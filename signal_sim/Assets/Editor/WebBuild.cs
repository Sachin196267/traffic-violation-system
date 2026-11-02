using System.IO;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.Rendering;

namespace SignalSim
{
    /// <summary>
    /// Browser build of the 3-D simulator, served by the web platform at /sim3d (app/server/main.py).
    /// The junction is built at runtime by SimView, so the build has one empty scene. Shaders looked up with
    /// Shader.Find and URP/Lit variants switched on from code are forced into the build (else pink or flat materials).
    /// Batch (needs the WebGL Build Support module): -executeMethod SignalSim.WebBuild.Run  -> signal_sim/Builds/WebGL
    /// </summary>
    public static class WebBuild
    {
        const string Scene = "Assets/Scenes/WebMain.unity";
        const string KeepDir = "Assets/Resources/WebGLKeep";

        public static void Run()
        {
            Directory.CreateDirectory("Assets/Scenes");
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            // the junction's primitives get colliders at runtime: an inactive object using each collider keeps the
            // physics module from being stripped out of the build
            var keep = new GameObject("KeepPhysics");
            keep.AddComponent<BoxCollider>(); keep.AddComponent<SphereCollider>(); keep.AddComponent<CapsuleCollider>(); keep.AddComponent<MeshCollider>();
            keep.SetActive(false);
            EditorSceneManager.SaveScene(scene, Scene);

            foreach (var n in new[] { "Skybox/Panoramic", "Skybox/Procedural", "SignalSim/Rain", "SignalSim/Text3D" })
                AlwaysInclude(Shader.Find(n), n);
            KeepVariants();
            AssetDatabase.SaveAssets();

            PlayerSettings.WebGL.compressionFormat = WebGLCompressionFormat.Disabled;   // served as plain files, no special headers
            PlayerSettings.WebGL.dataCaching = true;
            PlayerSettings.runInBackground = true;

            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { Scene },
                locationPathName = "Builds/WebGL",
                target = BuildTarget.WebGL,
                options = BuildOptions.None,
            });
            if (report.summary.result == BuildResult.Succeeded)
                File.WriteAllText("Builds/WebGL/index.html", Page);           // full-window page instead of the 960x600 template
            Debug.Log($"[webbuild] {report.summary.result}, {report.summary.totalSize / 1048576f:F1} MB, {report.summary.totalErrors} errors");
            if (Application.isBatchMode) EditorApplication.Exit(report.summary.result == BuildResult.Succeeded ? 0 : 1);
        }

        const string Page = @"<!DOCTYPE html>
<html lang=""en""><head><meta charset=""utf-8"">
<meta name=""viewport"" content=""width=device-width, initial-scale=1"">
<title>TraffiSense - 3D Junction Simulator</title>
<style>
html,body{margin:0;height:100%;overflow:hidden;background:#11161F;font-family:-apple-system,Segoe UI,Arial,sans-serif}
#c{width:100vw;height:100vh;display:block;background:#11161F}
#load{position:fixed;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:18px;color:#F3EFE6}
#load h1{margin:0;font-size:28px;font-weight:600;letter-spacing:.5px}
#load p{margin:0;color:#B9C3CF;font-size:15px}
#bar{width:320px;height:6px;background:#2C3646;border-radius:3px;overflow:hidden}
#fill{height:100%;width:0;background:#F2A33A;transition:width .2s}
#err{color:#ff8b7a;max-width:520px;text-align:center}
</style></head><body>
<canvas id=""c"" tabindex=""-1""></canvas>
<div id=""load""><h1>TraffiSense 3D Junction Simulator</h1><div id=""bar""><div id=""fill""></div></div>
<p>Loading the simulator (about 80 MB the first time)...</p><p id=""err""></p></div>
<script>
var s=document.createElement('script');s.src='Build/WebGL.loader.js';
s.onload=function(){createUnityInstance(document.getElementById('c'),{
 dataUrl:'Build/WebGL.data',frameworkUrl:'Build/WebGL.framework.js',codeUrl:'Build/WebGL.wasm',
 streamingAssetsUrl:'StreamingAssets',companyName:'TraffiSense',productName:'signal_sim',productVersion:'1.0'
},function(p){document.getElementById('fill').style.width=(100*p)+'%';})
.then(function(){document.getElementById('load').remove();document.getElementById('c').focus();})
.catch(function(e){document.getElementById('err').textContent='Could not start: '+e;});};
document.body.appendChild(s);
</script></body></html>";

        static void AlwaysInclude(Shader s, string name)
        {
            if (s == null) { Debug.LogWarning("[webbuild] shader not found: " + name); return; }
            var gs = AssetDatabase.LoadAssetAtPath<GraphicsSettings>("ProjectSettings/GraphicsSettings.asset");
            var so = new SerializedObject(gs);
            var arr = so.FindProperty("m_AlwaysIncludedShaders");
            for (int i = 0; i < arr.arraySize; i++)
                if (arr.GetArrayElementAtIndex(i).objectReferenceValue == s) return;
            arr.InsertArrayElementAtIndex(arr.arraySize);
            arr.GetArrayElementAtIndex(arr.arraySize - 1).objectReferenceValue = s;
            so.ApplyModifiedProperties();
        }

        /// <summary>one material per keyword combination Mats.cs / WorldBuilder.cs switch on, so the variants survive stripping</summary>
        static void KeepVariants()
        {
            Directory.CreateDirectory(KeepDir);
            var lit = Shader.Find("Universal Render Pipeline/Lit");
            string[] kw = { "_NORMALMAP", "_EMISSION", "_SMOOTHNESS_TEXTURE_ALBEDO_CHANNEL_A" };
            if (lit != null)
                for (int mask = 0; mask < 8; mask++)
                {
                    var m = new Material(lit);
                    for (int b = 0; b < kw.Length; b++) if ((mask & (1 << b)) != 0) m.EnableKeyword(kw[b]);
                    AssetDatabase.CreateAsset(m, $"{KeepDir}/lit_{mask}.mat");
                }
            foreach (var n in new[] { "SignalSim/Rain", "SignalSim/Text3D" })
            {
                var s = Shader.Find(n);
                if (s != null) AssetDatabase.CreateAsset(new Material(s), $"{KeepDir}/{n.Replace('/', '_')}.mat");
            }
        }
    }
}
