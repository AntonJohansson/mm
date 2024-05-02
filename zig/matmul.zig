const std = @import("std");

//
// Version 0
//

pub fn v0(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    _ = allocator;
    for (0..n) |i| {
        for (0..l) |j| {
            c[i*l + j] = 0;
            for (0..m) |k| {
                c[i*l + j] += a[i*m + k] * b[k*l + j];
            }
        }
    }
}

//
// Version 1
//

fn v1_inner(c: []f32, a: []f32, b: []f32, i: usize, m: u32, l: u32) void {
    for (0..l) |j| {
        c[i*l + j] = 0;
        for (0..m) |k| {
            c[i*l + j] += a[i*m + k] * b[k*l + j];
        }
    }
}

pub fn v1(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    var p: std.Thread.Pool = undefined;
    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v1_inner, .{c,a,b,i,m,l});
    }
    p.deinit();
}

//
// Version 2
//

fn v2_tr_inner(tb: []f32, b: []f32, i: usize, m: u32, l: u32) void {
    for (0..m) |j| {
        tb[i*m + j] = b[j*l + i];
    }
}

fn v2_inner(c: []f32, a: []f32, tb: []f32, i: usize, m: u32, l: u32) void {
    for (0..l) |j| {
        c[i*l + j] = 0;
        for (0..m) |k| {
            c[i*l + j] += a[i*m + k] * tb[j*m + k];
        }
    }
}

pub fn v2(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    var p: std.Thread.Pool = undefined;

    const tb = try allocator.alloc(f32, l*m);
    try p.init(.{.allocator = allocator});
    for (0..l) |i| {
        try p.spawn(v2_tr_inner, .{tb, b, i,m,l});
    }
    p.deinit();

    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v2_inner, .{c,a,tb,i,m,l});
    }
    p.deinit();
}

//
// Version 3
//

const pad_align = 8;

fn v3_pad_inner(pa: []f32, a: []f32, i: usize, pm: u32, m: u32) void {
    for (0..m) |j| {
        pa[i*pm + j] = a[i*m + j];
    }
}

fn v3_tr_inner(ptb: []f32, b: []f32, i: usize, l: u32, pm: u32, m: u32) void {
    for (0..m) |j| {
        ptb[i*pm + j] = b[j*l + i];
    }
}

fn v3_mul_inner(c: []f32, pa: []f32, ptb: []f32, i: usize, l: u32, ma: u32, pm: u32) void {
    for (0..l) |j| {
        var d = [_]f32 {0}**pad_align;

        for (0..ma) |ka| {
            for (0..pad_align) |kb| {
                d[kb] += pa[i*pm + ka*pad_align + kb] * ptb[j*pm + ka*pad_align + kb];
            } }
        c[i*l + j] = 0;
        for (0..pad_align) |k| {
            c[i*l + j] += d[k];
        }
    }
}

pub fn v3(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    const ma = (m + pad_align - 1) / pad_align;
    const la = (l + pad_align - 1) / pad_align;
    const pm = ma*pad_align;
    const pl = la*pad_align;

    const pa  = try allocator.alloc(f32, n*pm);
    const ptb = try allocator.alloc(f32, pl*pm);
    @memset(pa, 0);
    @memset(ptb, 0);

    var p: std.Thread.Pool = undefined;

    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v3_pad_inner, .{pa, a, i, pm ,m});
    }
    p.deinit();

    try p.init(.{.allocator = allocator});
    for (0..l) |i| {
        try p.spawn(v3_tr_inner, .{ptb, b, i, l, pm, m});
    }
    p.deinit();

    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v3_mul_inner, .{c, pa, ptb, i, l, ma, pm});
    }
    p.deinit();
}

//
// Version 4
//

fn v4_mul_inner(c: []f32, d: []f32, pa: []f32, ptb: []f32, l: u32, ma: u32, i: usize, pm: u32) void {
    for (0..l) |j| {
        var vd: @Vector(8, f32) = d[0..8].*;

        for (0..ma) |ka| {
            const base_a  = i*pm + ka*pad_align;
            const base_b = j*pm + ka*pad_align;
            const va: @Vector(8, f32) =  pa[base_a..(base_a+8)][0..8].*;
            const vb: @Vector(8, f32) = ptb[base_b..(base_b+8)][0..8].*;
            vd += va*vb;
        }

        c[i*l + j] = 0;
        for (0..8) |kb| {
            c[i*l + j] += vd[kb];
        }
    }
}

pub fn v4(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    const ma = (m + pad_align - 1) / pad_align;
    const la = (l + pad_align - 1) / pad_align;
    const pm = ma*pad_align;
    const pl = la*pad_align;

    const pa  = try allocator.alloc(f32, n*pm);
    const ptb = try allocator.alloc(f32, pl*pm);
    @memset(pa, 0);
    @memset(ptb, 0);

    var p: std.Thread.Pool = undefined;

    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v3_pad_inner, .{pa, a, i, pm ,m});
    }
    p.deinit();

    try p.init(.{.allocator = allocator});
    for (0..l) |i| {
        try p.spawn(v3_tr_inner, .{ptb, b, i, l, pm, m});
    }
    p.deinit();

    const d = try allocator.allocWithOptions(f32, pad_align*@sizeOf(f32), pad_align*@sizeOf(f32), null);
    @memset(d, 0);

    try p.init(.{.allocator = allocator});
    for (0..n) |i| {
        try p.spawn(v4_mul_inner, .{c, d, pa, ptb, l, ma, i, pm});
    }
    p.deinit();
}

//void matmul_v5(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + pad_align - 1) / pad_align;
//    const u32 la = (l + pad_align - 1) / pad_align;
//    const u32 pm = ma*pad_align;
//    const u32 pl = la*pad_align;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * n * pm);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(1, sizeof(f32) * pl * pm);
//    memset(pa,  0, sizeof(f32) * n * pm);
//    memset(ptb, 0, sizeof(f32) * pl * pm);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//    f32 *d = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32));
//    memset(d, 0, sizeof(f32)*8);
//
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < l; ++j) {
//            __m256 vd = _mm256_loadu_ps(&d[0]);
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                __m256 va = _mm256_loadu_ps( &pa[i*pm + ka*pad_align]);
//                __m256 vb = _mm256_loadu_ps(&ptb[j*pm + ka*pad_align]);
//                vd = _mm256_fmadd_ps(va, vb, vd);
//            }
//
//            c[i*l + j] = hsum256_ps_avx(vd);
//        }
//    }
//}
//
//inline f32 hadd(__m256 vd, u32 nb) {
//    f32 sum = 0;
//    for (int kb = 0; kb < nb; ++kb)
//        sum += vd[kb];
//    return sum;
//}
//
//#define NB 8
//#define ND 3
//
//void matmul_v6(f32 * restrict c,
//               const f32 * restrict a,
//               const f32 * restrict b,
//               u32 n, u32 m, u32 l) {
//    const u32 ma = (m + NB - 1) / NB;
//    const u32 la = (l + NB - 1) / NB;
//    const u32 pm = ma*NB;
//    const u32 pl = la*NB;
//
//    const u32 nc = (n + ND - 1) / ND;
//    const u32 ncd = nc*ND;
//
//    const u32 mc = (m + ND - 1) / ND;
//    const u32 mcd = mc*ND;
//
//    f32 *pa  = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*ncd*ma);
//    f32 *ptb = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*mcd*la);
//    memset(pa, 0, sizeof(f32)*8 * ncd * ma);
//    memset(ptb, 0, sizeof(f32)*8 * mcd * la);
//#pragma omp parallel for
//    for (u32 i = 0; i < n; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            pa[i*pm + j] = a[i*m + j];
//        }
//    }
//#pragma omp parallel for
//    for (u32 i = 0; i < l; ++i) {
//        for (u32 j = 0; j < m; ++j) {
//            ptb[i*pm + j] = b[j*l + i];
//        }
//    }
//
//    f32 *d = MM_ALLOCATE_ALIGNED(8*sizeof(f32), 8*sizeof(f32)*ND*ND);
//    memset(d, 0, sizeof(f32)*8*ND*ND);
//
//#pragma omp parallel for
//    for (u32 ic = 0; ic < nc; ++ic) {
//        for (u32 jc = 0; jc < nc; ++jc) {
//            __m256 vd00 = _mm256_loadu_ps(&d[0*ND + 0]);
//            __m256 vd01 = _mm256_loadu_ps(&d[0*ND + 1]);
//            __m256 vd02 = _mm256_loadu_ps(&d[0*ND + 2]);
//            __m256 vd10 = _mm256_loadu_ps(&d[1*ND + 0]);
//            __m256 vd11 = _mm256_loadu_ps(&d[1*ND + 1]);
//            __m256 vd12 = _mm256_loadu_ps(&d[1*ND + 2]);
//            __m256 vd20 = _mm256_loadu_ps(&d[2*ND + 0]);
//            __m256 vd21 = _mm256_loadu_ps(&d[2*ND + 1]);
//            __m256 vd22 = _mm256_loadu_ps(&d[2*ND + 2]);
//
//            for (u32 ka = 0; ka < ma; ++ka) {
//                __m256 va0 = _mm256_loadu_ps( &pa[(ic*ND + 0)*ma*NB + ka*NB]);
//                __m256 va1 = _mm256_loadu_ps( &pa[(ic*ND + 1)*ma*NB + ka*NB]);
//                __m256 va2 = _mm256_loadu_ps( &pa[(ic*ND + 2)*ma*NB + ka*NB]);
//                __m256 vb0 = _mm256_loadu_ps(&ptb[(jc*ND + 0)*ma*NB + ka*NB]);
//                __m256 vb1 = _mm256_loadu_ps(&ptb[(jc*ND + 1)*ma*NB + ka*NB]);
//                __m256 vb2 = _mm256_loadu_ps(&ptb[(jc*ND + 2)*ma*NB + ka*NB]);
//                vd00 = _mm256_fmadd_ps(va0, vb0, vd00);
//                vd01 = _mm256_fmadd_ps(va0, vb1, vd01);
//                vd02 = _mm256_fmadd_ps(va0, vb2, vd02);
//                vd10 = _mm256_fmadd_ps(va1, vb0, vd10);
//                vd11 = _mm256_fmadd_ps(va1, vb1, vd11);
//                vd12 = _mm256_fmadd_ps(va1, vb2, vd12);
//                vd20 = _mm256_fmadd_ps(va2, vb0, vd20);
//                vd21 = _mm256_fmadd_ps(va2, vb1, vd21);
//                vd22 = _mm256_fmadd_ps(va2, vb2, vd22);
//            }
//
//            u32 i, j;
//            i = ic*ND + 0; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd00, NB);
//            i = ic*ND + 0; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd01, NB);
//            i = ic*ND + 0; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd02, NB);
//            i = ic*ND + 1; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd10, NB);
//            i = ic*ND + 1; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd11, NB);
//            i = ic*ND + 1; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd12, NB);
//            i = ic*ND + 2; j = jc*ND + 0; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd20, NB);
//            i = ic*ND + 2; j = jc*ND + 1; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd21, NB);
//            i = ic*ND + 2; j = jc*ND + 2; if (i < n && j < n) c[(i)*l + (j)] = hadd(vd22, NB);
//        }
//    }
//}

