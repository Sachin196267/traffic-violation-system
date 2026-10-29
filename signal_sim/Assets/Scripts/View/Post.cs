using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Rendering.Universal;

namespace SignalSim
{
    /// <summary>Film-like post-processing (URP): ACES tone mapping, bloom, grading, vignette, depth of field when following.</summary>
    public static class Post
    {
        static Volume vol;
        static Bloom bloom;
        static ColorAdjustments grade;
        static DepthOfField dof;

        public static void Make(Transform parent)
        {
            if (!Mats.Urp) return;
            var go = new GameObject("Post-processing");
            go.transform.SetParent(parent, false);
            vol = go.AddComponent<Volume>();
            vol.isGlobal = true;
            var p = ScriptableObject.CreateInstance<VolumeProfile>();
            p.Add<Tonemapping>(true).mode.Override(TonemappingMode.ACES);
            bloom = p.Add<Bloom>(true);
            bloom.threshold.Override(1.0f); bloom.intensity.Override(0.45f); bloom.scatter.Override(0.68f);
            grade = p.Add<ColorAdjustments>(true);
            grade.contrast.Override(14f); grade.saturation.Override(6f); grade.postExposure.Override(0.25f);
            var vg = p.Add<Vignette>(true); vg.intensity.Override(0.22f); vg.smoothness.Override(0.45f);
            p.Add<WhiteBalance>(true).temperature.Override(7f);   // warm, dusty Indian afternoon
            var smh = p.Add<ShadowsMidtonesHighlights>(true);
            smh.shadows.Override(new Vector4(0.95f, 0.98f, 1.06f, 0f));   // cool shadows, warm light
            dof = p.Add<DepthOfField>(true);
            dof.mode.Override(DepthOfFieldMode.Off);
            vol.sharedProfile = p;
        }

        public static void SetupCamera(Camera c, bool post)
        {
            if (!Mats.Urp) return;
            var d = c.GetUniversalAdditionalCameraData();
            d.renderPostProcessing = post;
            d.antialiasing = AntialiasingMode.SubpixelMorphologicalAntiAliasing;
            d.antialiasingQuality = AntialiasingQuality.High;
            d.renderShadows = true;
            d.requiresDepthOption = CameraOverrideOption.On;   // depth of field and ambient occlusion read scene depth
        }

        public static void TimeOfDay(float night)
        {
            if (vol == null) return;
            bloom.intensity.Override(Mathf.Lerp(0.45f, 1.1f, night));
            grade.postExposure.Override(Mathf.Lerp(0.25f, 0.6f, night));
            grade.saturation.Override(Mathf.Lerp(6f, -8f, night));
        }

        /// <summary>focus on a followed vehicle (blurs the background); distance &lt;= 0 switches it off</summary>
        public static void Focus(float distance)
        {
            if (dof == null) return;
            if (distance <= 0f) { dof.mode.Override(DepthOfFieldMode.Off); return; }
            dof.mode.Override(DepthOfFieldMode.Bokeh);
            dof.focusDistance.Override(distance);
            dof.focalLength.Override(55f);
            dof.aperture.Override(4.5f);
        }
    }
}
