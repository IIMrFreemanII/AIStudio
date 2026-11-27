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

namespace orb {
constant float3 spark = float3(0.66, 0.74, 1.0);

float hash(float n) { return fract(sin(n) * 43758.5453); }

float2 rotate(float2 v, float a) { float c = cos(a), s = sin(a); return float2(c * v.x - s * v.y, s * v.x + c * v.y); }

// Paints `c` with alpha `a` over `acc` (premultiplied).
void over(thread float4 &acc, float3 c, float a) {
    a = clamp(a, 0.0, 1.0);
    acc.rgb = c * a + acc.rgb * (1.0 - a);
    acc.a = a + acc.a * (1.0 - a);
}
}

/// What goes around the sphere, in depth: the architect's tilted ring and its spark, and the sparkles orbiting it.
/// Drawn twice on a square 2.4 radii across: once behind the sphere (`front` 0: what's further away, which the sphere
/// hides) and once over it (`front` 1).
/// - box: the layer's bounding rect; scale: pixels per point; extent: the box's half-width in sphere radii.
/// - ring: shown (0/1), the spark's angle, the ring's tilt; particles: how many (0–8); seed: per actor.
/// - time: seconds (wrapped by the caller); lightWindow: 1 on a light window (a violet ring, pure-colour sparkles).
[[ stitchable ]] half4 studioOrbAround(float2 position, half4 color, float4 box, float scale, float extent, float front,
                                       float3 tint, float3 rimColor, float architect,
                                       float3 ring, float particles, float seed, float time, float lightWindow) {
    float half_ = min(box.z, box.w) * 0.5;
    float r = half_ / extent;
    float2 center = box.xy + box.zw * 0.5;
    float2 p = (position - center) / r;
    p.y = -p.y;
    float aa = 1.0 / max(r * scale, 1.0);
    bool wantFront = front > 0.5;
    bool light = lightWindow > 0.5;
    float4 acc = float4(0.0);

    // The ring: a circle 1.62 radii across, tilted so it looks like an ellipse 0.44 tall; its lower half is nearer.
    if (ring.x > 0.5) {
        float2 q = orb::rotate(p, -ring.z);
        float A = 1.62, B = 0.44;
        float l = length(q / float2(A, B));
        float g = length(q / float2(A * A, B * B)) / max(l, 1e-4);
        float d = abs(l - 1.0) / max(g, 1e-4);
        float3 rc = light ? orb::violet : rimColor;
        float ra = light ? 0.7 : 0.55;
        float lw = max(light ? 0.056 : 0.044, 0.6 / r);
        if ((q.y < 0.0) == wantFront) orb::over(acc, rc, ra * (1.0 - smoothstep(lw * 0.5 - aa, lw * 0.5 + aa, d)));
        // The spark rides it clockwise; nearer, it's a little bigger.
        float th = ring.y;
        if ((sin(th) > 0.0) == wantFront) {
            float2 sp = float2(A * cos(th), -B * sin(th));
            float sc = max(0.22 * (1.0 + 0.3 * sin(th)), 2.0 / r);
            float2 s = (q - sp) / sc;
            float ds = sqrt(abs(s.x)) + sqrt(abs(s.y)) - 1.0;
            float aaS = aa / sc * 2.0;
            orb::over(acc, orb::spark, exp(-dot(s, s) * 1.6) * 0.75);
            orb::over(acc, light ? orb::violet : float3(1.0), 1.0 - smoothstep(-aaS, aaS, ds));
        }
    }

    // Sparkles on tilted orbits 1.3–2 radii out, fading in and out, in the orb's own colours.
    for (int i = 0; i < 8; i++) {
        if (float(i) >= particles) break;
        float fi = float(i) + seed * 13.0;
        float rad = 1.3 + 0.75 * orb::hash(fi * 1.7);
        float spd = (0.22 + 0.3 * orb::hash(fi * 2.3)) * (orb::hash(fi * 5.1) > 0.5 ? 1.0 : -1.0);
        float ph = orb::hash(fi * 3.9) * 6.283;
        float incl = (orb::hash(fi * 7.3) - 0.5) * 2.2;
        float az = orb::hash(fi * 9.1) * 6.283;
        float a = ph + time * spd;
        float3 v = float3(cos(a), 0.0, sin(a)) * rad;
        v = float3(v.x, -v.z * sin(incl), v.z * cos(incl));
        v = float3(cos(az) * v.x + sin(az) * v.z, v.y, -sin(az) * v.x + cos(az) * v.z);
        if ((v.z > 0.0) != wantFront) continue;
        float life = 0.5 + 0.5 * sin(time * (0.6 + 0.5 * orb::hash(fi * 4.4)) + ph * 2.0);
        float sz = max(0.07 * (1.0 + 0.3 * v.z / rad), 0.8 / r);
        float d = length(p - v.xy) - sz;
        float gd = max(d, 0.0);
        float3 pc = architect > 0.5 ? orb::iridescent(orb::hash(fi * 6.7)) : tint;
        pc = mix(pc, float3(1.0), light ? 0.0 : 0.3);
        orb::over(acc, pc, exp(-gd * gd / (sz * sz * 4.0)) * 0.35 * life);
        orb::over(acc, pc, (1.0 - smoothstep(-aa, aa, d)) * 0.95 * life);
    }

    return half4(acc);
}
