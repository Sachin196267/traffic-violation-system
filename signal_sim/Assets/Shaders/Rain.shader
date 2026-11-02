// Rain streaks: soft additive, depth-tested, no lighting (works in URP and built-in)
Shader "SignalSim/Rain"
{
    Properties { _Color ("Color", Color) = (0.7,0.75,0.8,0.25) }
    SubShader
    {
        Tags { "Queue"="Transparent" "IgnoreProjector"="True" "RenderType"="Transparent" }
        Lighting Off Cull Off ZWrite Off ZTest LEqual
        Blend SrcAlpha One
        Pass
        {
            CGPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #include "UnityCG.cginc"
            struct appdata { float4 vertex : POSITION; fixed4 color : COLOR; float2 uv : TEXCOORD0; };
            struct v2f { float4 pos : SV_POSITION; fixed4 color : COLOR; float2 uv : TEXCOORD0; };
            fixed4 _Color;
            v2f vert (appdata v) { v2f o; o.pos = UnityObjectToClipPos(v.vertex); o.color = v.color * _Color; o.uv = v.uv; return o; }
            fixed4 frag (v2f i) : SV_Target { fixed4 c = i.color; c.a *= saturate(1 - abs(i.uv.x - 0.5) * 2); return c; }
            ENDCG
        }
    }
}
