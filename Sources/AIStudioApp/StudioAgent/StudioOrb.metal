// The 3D studio orb: a sphere of glossy dark glass, shaded per pixel (a SwiftUI colorEffect on a square the orb's size).
// Ported from the mock-up page (https://claude.ai/artifact/51wJkptJqTgM8kBFCnBTgz). The halo, the architect's ring and
// the particles are still drawn by StudioOrb around it.
//
// Coordinates: p is in sphere radii with y up; the view looks down -z, so the surface's normal is (p, sqrt(1 - p·p)).
// `rot` turns the body (eyes, inner glow) into view space; the light stays fixed, so the highlight doesn't move.

#include <metal_stdlib>
#include <SwiftUI/SwiftUI_Metal.h>
using namespace metal;

namespace orb {
constant float3 glass = float3(0.071, 0.063, 0.098);
constant float3 eye = float3(0.93, 0.92, 0.98);
constant float3 violet = float3(0.50, 0.47, 0.87);
constant float3 magenta = float3(0.83, 0.33, 0.53);
constant float3 teal = float3(0.36, 0.79, 0.65);

float3 spin(float3 v, float a) {
    float3 k = normalize(float3(0.25, 1.0, 0.15));
    float c = cos(a), s = sin(a);
    return v * c + cross(k, v) * s + k * dot(k, v) * (1.0 - c);
}

float3 iridescent(float x) {
    x = fract(x);
    float3 c = mix(violet, magenta, smoothstep(0.0, 0.33, x));
    c = mix(c, teal, smoothstep(0.33, 0.66, x));
    return mix(c, violet, smoothstep(0.66, 1.0, x));
}

// Glow from a point inside the glass, seen along the view ray through the sphere's thickness.
float glowAt(float2 p, float zf, float3 g, float sigma) {
    float dz = max(abs(g.z) - zf, 0.0);
    float2 d = p - g.xy;
    return exp(-(dot(d, d) + dz * dz) / (sigma * sigma));
}

float sdPill(float2 p, float2 wh) {
    float r = min(wh.x, wh.y) * 0.5;
    float2 q = abs(p) - (wh * 0.5 - r);
    return length(max(q, 0.0)) + min(max(q.x, q.y), 0.0) - r;
}

// ^ : an arc around +y.
float sdHappy(float2 p) {
    p.y += 0.09;
    float2 sc = float2(sin(1.15), cos(1.15));
    p.x = abs(p.x);
    float ra = 0.17;
    float d = (sc.y * p.x > sc.x * p.y) ? length(p - sc * ra) : abs(length(p) - ra);
    return d - 0.091;
}

float sdSegment(float2 p, float2 a, float2 b) {
    float2 pa = p - a, ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h);
}

float sdCross(float2 p) {
    float a = 0.13;
    return min(sdSegment(p, float2(-a), float2(a)), sdSegment(p, float2(-a, a), float2(a, -a))) - 0.073;
}
}

/// - box: the layer's bounding rect (x, y, width, height), in points.
/// - scale: the display's pixels per point (for antialiasing).
/// - rot / rotCount: the body's 3×3 rotation, column by column.
/// - eye: shape (0 pill, 1 happy, 2 cross), width, height, lift; eyeGap: between the eye centres.
/// - glow: the inner glow's brightness (breathing, paused); spin: the inner glow's drift angle.
/// - glass: highlight size, highlight strength, rim, frost; glass2: glow spread, glow strength, iridescence.
[[ stitchable ]] half4 studioOrbSphere(float2 position, half4 color, float4 box, float scale,
                                       float3 tint, float3 rimColor, float architect,
                                       device const float *rot, int rotCount,
                                       float4 eye, float eyeGap, float glow, float spin,
                                       float4 glass, float3 glass2) {
    float r = min(box.z, box.w) * 0.5;
    float2 center = box.xy + box.zw * 0.5;
    float2 p = (position - center) / r;
    p.y = -p.y;
    float aa = 1.0 / max(r * scale, 1.0);
    float rl = length(p);
    float alpha = 1.0 - smoothstep(1.0 - aa, 1.0 + aa, rl);
    if (alpha <= 0.0 || rotCount < 9) return half4(0);

    float3x3 body = float3x3(float3(rot[0], rot[1], rot[2]), float3(rot[3], rot[4], rot[5]), float3(rot[6], rot[7], rot[8]));
    float2 pp = rl > 0.999 ? p / rl * 0.999 : p;
    float zf = sqrt(max(1.0 - dot(pp, pp), 0.0));
    float3 n = float3(pp, zf);
    float3 nb = transpose(body) * n;

    // Dark glass, faintly lighter at the top, with the glow inside it (low and right; the architect has three).
    float gs = glass2.y * glow;
    float sigma = glass2.x;
    float3 col = orb::glass * (0.8 + 0.35 * n.y);
    if (architect > 0.5) {
        col += orb::magenta * 0.8 * gs * orb::glowAt(pp, zf, body * orb::spin(float3(0.42, -0.48, 0.3), spin), sigma);
        col += orb::violet * 0.55 * gs * orb::glowAt(pp, zf, body * orb::spin(float3(-0.42, 0.44, 0.25), spin), sigma * 0.85);
        col += orb::teal * 0.3 * gs * orb::glowAt(pp, zf, body * orb::spin(float3(0.55, 0.08, -0.2), spin), sigma * 0.8);
    } else {
        col += tint * 0.85 * gs * orb::glowAt(pp, zf, body * orb::spin(float3(0.42, -0.48, 0.3), spin), sigma);
    }

    // Eyes, painted on the surface in the body's frame: they turn with it and foreshorten toward the edge.
    float dmin = 1e3;
    for (int i = 0; i < 2; i++) {
        float side = i == 0 ? -1.0 : 1.0;
        float2 e2 = float2(-0.2 + side * eyeGap * 0.5, 0.12 + eye.w);
        float3 c = normalize(float3(e2, sqrt(max(1.0 - dot(e2, e2), 0.0))));
        float3 tu = normalize(float3(c.z, 0.0, -c.x));
        float3 tv = cross(c, tu);
        float dc = dot(nb, c);
        if (dc < 0.25) continue;
        float2 uv = float2(dot(nb, tu), dot(nb, tv)) / dc;
        float d;
        if (eye.x < 0.5) d = orb::sdPill(uv, eye.yz);
        else if (eye.x < 1.5) d = orb::sdHappy(uv);
        else d = orb::sdCross(uv);
        dmin = min(dmin, d);
    }
    float aaEye = aa * 1.2 / max(nb.z * 0.5 + n.z * 0.5, 0.2);
    float eyeAlpha = 1.0 - smoothstep(-aaEye, aaEye, dmin);
    col += orb::eye * 0.16 * exp(-max(dmin, 0.0) * 16.0) * (1.0 - eyeAlpha);
    col = mix(col, orb::eye, eyeAlpha);

    // Rim: colour from the viewing angle (fresnel) plus a frosted band, iridescent on the architect.
    float fres = pow(1.0 - zf, 2.2);
    float frost = smoothstep(1.0 - max(glass.w, 0.001), 1.0, rl);
    float3 rim = rimColor;
    if (architect > 0.5) {
        float3 film = orb::iridescent(fres * 0.9 + atan2(nb.y, nb.x) * 0.16 + spin * 0.12);
        rim = mix(rim, film, glass2.z);
        col += film * 0.1 * glass2.z * fres;
    }
    col = mix(col, rim, clamp(glass.z * (0.55 * fres + 0.7 * frost), 0.0, 0.95));

    // Gloss: a sharp highlight up and left near the rim (clear of the eyes), a soft sheen, and a thin bright arc along
    // the upper-left edge.
    float3 h = normalize(float3(-0.56, 0.6, 0.57));
    float ndh = clamp(dot(n, h), -1.0, 1.0);
    float ang = acos(ndh);
    float core = 1.0 - smoothstep(glass.x * 0.55, glass.x, ang);
    float soft = pow(max(ndh, 0.0), 40.0) * 0.3;
    float2 dir = pp / max(length(pp), 1e-4);
    float arc = pow(fres, 2.5) * pow(max(dot(dir, normalize(float2(-0.6, 0.8))), 0.0), 6.0);
    col = mix(col, float3(1.0), clamp(glass.y * (0.95 * core + soft + 0.65 * arc), 0.0, 1.0));

    return half4(half3(col * alpha), half(alpha));
}
