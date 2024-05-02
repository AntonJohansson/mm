const std = @import("std");
const mm = @import("matmul.zig");

var prng = std.rand.DefaultPrng.init(0);
const rand = prng.random();

const stdout_file = std.io.getStdOut().writer();
var bw = std.io.bufferedWriter(stdout_file);
const stdout = bw.writer();

fn logf(comptime format: []const u8, args: anytype) void {
    stdout.print(format, args) catch unreachable;
    bw.flush() catch unreachable;
}

fn log(comptime msg: []const u8) void {
    stdout.writeAll(msg) catch unreachable;
    bw.flush() catch unreachable;
}

pub fn main() !void {
    var gpa_state = std.heap.GeneralPurposeAllocator(.{}){};
    const gpa = gpa_state.allocator();

    const memory_size = 1 << 31;
    const memory = try gpa.alloc(u8, memory_size);
    defer gpa.free(memory);
    var stack_state = std.heap.FixedBufferAllocator.init(memory);
    const stack = stack_state.allocator();

    const step = 10;
    const max_size = 800;
    logf("{s:10}", .{"n"});
    logf("{s:16}", .{"v0"});
    logf("{s:16}", .{"v1"});
    logf("{s:16}", .{"v2"});
    logf("{s:16}", .{"v3"});
    logf("{s:16}", .{"v4"});
    //logf("{s:16}", .{"v5"});
    //logf("{s:16}", .{"v6"});
    log("\n");

    var i: usize = step;
    while (i <= max_size) : (i += step) {
        const n: u32 = @intCast(i);
        const m: u32 = @intCast(i);
        const l: u32 = @intCast(i);

        const a = try stack.alloc(f32, n * m);
        const b = try stack.alloc(f32, m * l);
        const c = try stack.alloc(f32, n * l);
        const d = try stack.alloc(f32, n * l);

        random_matrix(a, n,m);
        random_matrix(b, m,l);

        logf("{:10}", .{i});
        //try bench_matmul(stack, mm.v0, iterations, c,a,b, n,m,l); @memcpy(d, c);
        {
            const before = rdtscp();
            try mm.v0(stack, c,a,b, n,m,l);
            const after = rdtscp();
            logf("{:16}", .{after-before});
            @memcpy(d, c);
        }
        {
            const before = rdtscp();
            try mm.v1(stack, c,a,b, n,m,l);
            const after = rdtscp();
            logf("{:16}", .{after-before});
            assert(matrix_equal(d,c,n,l));
        }
        {
            const before = rdtscp();
            try mm.v2(stack, c,a,b, n,m,l);
            const after = rdtscp();
            logf("{:16}", .{after-before});
            assert(matrix_equal(d,c,n,l));
        }
        {
            const before = rdtscp();
            try mm.v3(stack, c,a,b, n,m,l);
            const after = rdtscp();
            logf("{:16}", .{after-before});
            assert(matrix_equal(d,c,n,l));
        }
        {
            const before = rdtscp();
            try mm.v4(stack, c,a,b, n,m,l);
            const after = rdtscp();
            logf("{:16}", .{after-before});
            assert(matrix_equal(d,c,n,l));
        }
        //try bench_matmul(stack, mm.v1, iterations, c,a,b, n,m,l); 
        //try bench_matmul(stack, mm.v2, iterations, c,a,b, n,m,l); std.debug.assert(matrix_equal(d,c,n,l));
        log("\n");

        stack_state.reset();
    }

    //    bench_matmul(matmul_v0, iterations, c,a,b, n,m,l); memcpy(d, c, sizeof(f32)*n*l);
    //    bench_matmul(matmul_v1, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v2, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v3, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v4, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v5, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(matmul_v6, iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));
    //    bench_matmul(gemm,      iterations, c,a,b, n,m,l); assert(matrix_equal(d,c, n,l));

    //    stack.top = 0;

    //    printf("\n");
    //}

    try bw.flush();
}

fn random_matrix(mat: []f32, n: u32, m: u32) void {
    for (0..n) |i| {
        for (0..m) |j| {
            mat[i*m + j] = 1.0 - 2.0 * (rand.float(f32));
        }
    }
}

fn matrix_equal(a: []f32, b: []f32, n: u32, m: u32) bool {
    for (0..n) |i| {
        for (0..m) |j| {
            if (!std.math.approxEqAbs(f32, a[i*m+j], b[i*m+j], 1e-4)) {
                logf("\n{} vs {}\n", .{a[i*m+j], b[i*m+j]});
                return false;
            }
        }
    }
    return true;
}

const MmFunc = *const fn(allocator: std.mem.Allocator, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) std.mem.Allocator.Error!void;
fn bench_matmul(allocator: std.mem.Allocator, mmfunc: MmFunc, iterations: usize, c: []f32, a: []f32, b: []f32, n: u32, m: u32, l: u32) !void {
    const before = rdtscp();
    for (0..iterations) |_| {
        try mmfunc(allocator, c, a, b, n,m,l);
    }
    const after = rdtscp();
    logf("{:16}", .{after-before});
}

fn rdtscp() u64 {
    return asm volatile (
        \\rdtscp;
        \\shl $32, %rdx;
        \\or %rdx, %rax
        : [ret] "={rax}" (-> u64)
        :
        : "%rcx", "%rdx"
    );
}

fn assert(expr: bool) void {
    if (!expr) {
        log("assert failed!\n");
    }
}
