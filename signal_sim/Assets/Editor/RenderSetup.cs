using System;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace SignalSim
{
    /// <summary>
    /// One-time project setup for realistic rendering: Universal Render Pipeline (HDR, 4x MSAA, 4 shadow cascades out to
    /// 180 m, 4K soft sun shadows, screen-space ambient occlusion) and linear colour space. Runs automatically when the
    /// editor loads; batch: -executeMethod SignalSim.RenderSetup.Ensure
    /// </summary>
    [InitializeOnLoad]
    public static class RenderSetup
    {
        const string Dir = "Assets/Settings";
        const string PipelinePath = Dir + "/SignalSim_URP.asset", RendererPath = Dir + "/SignalSim_Renderer.asset";

        static RenderSetup() { EditorApplication.delayCall += () => Ensure(); }

        [MenuItem("Signal Sim/Set up realistic rendering (URP)")]
        public static void Ensure()
        {
            if (PlayerSettings.colorSpace != ColorSpace.Linear) PlayerSettings.colorSpace = ColorSpace.Linear;
            var asset = AssetDatabase.LoadAssetAtPath<UniversalRenderPipelineAsset>(PipelinePath);
            if (asset == null)
            {
                Directory.CreateDirectory(Dir);
                var rd = ScriptableObject.CreateInstance<UniversalRendererData>();
                AssetDatabase.CreateAsset(rd, RendererPath);
                AddFeature(rd, "UnityEngine.Rendering.Universal.ScreenSpaceAmbientOcclusion", "SSAO");
                asset = UniversalRenderPipelineAsset.Create(rd);
                AssetDatabase.CreateAsset(asset, PipelinePath);
                asset.shadowDistance = 180f;
                asset.shadowCascadeCount = 4;
                asset.msaaSampleCount = 4;
                asset.supportsHDR = true;
                asset.maxAdditionalLightsCount = 8;
                var so = new SerializedObject(asset);
                Set(so, "m_MainLightShadowmapResolution", 4096);
                Set(so, "m_SoftShadowsSupported", true);
                Set(so, "m_AdditionalLightShadowsSupported", false);
                Set(so, "m_SupportsCameraDepthTexture", true);
                Set(so, "m_SupportsCameraOpaqueTexture", true);
                so.ApplyModifiedPropertiesWithoutUndo();
                EditorUtility.SetDirty(asset);
            }
            if (GraphicsSettings.defaultRenderPipeline != asset) GraphicsSettings.defaultRenderPipeline = asset;
            int cur = QualitySettings.GetQualityLevel();
            for (int i = 0; i < QualitySettings.names.Length; i++)
            {
                QualitySettings.SetQualityLevel(i, false);
                if (QualitySettings.renderPipeline != asset) QualitySettings.renderPipeline = asset;
            }
            QualitySettings.SetQualityLevel(cur, false);
            AssetDatabase.SaveAssets();
            Debug.Log("[render] URP ready: " + PipelinePath);
        }

        static void AddFeature(UniversalRendererData rd, string typeName, string name)
        {
            var t = Type.GetType(typeName + ", Unity.RenderPipelines.Universal.Runtime");
            if (t == null) { Debug.LogWarning("[render] renderer feature not found: " + typeName); return; }
            var f = (ScriptableRendererFeature)ScriptableObject.CreateInstance(t);
            f.name = name;
            AssetDatabase.AddObjectToAsset(f, rd);
            AssetDatabase.TryGetGUIDAndLocalFileIdentifier(f, out _, out long id);
            var so = new SerializedObject(rd);
            var list = so.FindProperty("m_RendererFeatures");
            var map = so.FindProperty("m_RendererFeatureMap");
            list.InsertArrayElementAtIndex(list.arraySize);
            list.GetArrayElementAtIndex(list.arraySize - 1).objectReferenceValue = f;
            if (map != null) { map.InsertArrayElementAtIndex(map.arraySize); map.GetArrayElementAtIndex(map.arraySize - 1).longValue = id; }
            so.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(rd);
        }

        static void Set(SerializedObject so, string prop, object v)
        {
            var p = so.FindProperty(prop);
            if (p == null) { Debug.LogWarning("[render] no setting " + prop); return; }
            if (v is bool b) p.boolValue = b; else if (v is int n) p.intValue = n; else if (v is float x) p.floatValue = x;
        }
    }

    /// <summary>import settings for the Poly Haven textures and skies in Resources/PH</summary>
    class PolyHavenImport : AssetPostprocessor
    {
        void OnPreprocessTexture()
        {
            if (!assetPath.Contains("/Resources/PH/")) return;
            var ti = (TextureImporter)assetImporter;
            ti.anisoLevel = 8;
            ti.wrapMode = TextureWrapMode.Repeat;
            ti.maxTextureSize = 2048;
            if (assetPath.EndsWith("_normal.png")) ti.textureType = TextureImporterType.NormalMap;
            else if (assetPath.EndsWith("_albedo.png")) { ti.sRGBTexture = true; ti.alphaSource = TextureImporterAlphaSource.FromInput; ti.alphaIsTransparency = false; }
            else if (assetPath.EndsWith(".hdr")) { ti.mipmapEnabled = false; ti.wrapModeU = TextureWrapMode.Repeat; ti.wrapModeV = TextureWrapMode.Clamp; }
        }
    }
}
