"""C++ self-checking csim testbench emitter for the `generic` Vitis target.

Deterministic small-integer stimulus (exactly representable, no overflow for the
default precisions), a double golden reference, and a 0.5 tolerance compare. The
top's return code is what Vitis ``csim_design`` checks: 0 = pass.

For const_weights packages the golden reads the same baked ROM the IP uses
(``<name>_weight_cols_rom`` from ``<name>_weights.h``), so it matches by
construction regardless of the baked values.
"""


def _prototype(name, m, k, n, interface, weights_in_core):
    if interface == "array" and not weights_in_core:
        return (f"void {name}({name}_a_row_t a_rows[{m}], {name}_b_col_t b_cols[{n}], "
                f"{name}_res_row_t results[{m}]);")
    if interface == "array" and weights_in_core:
        return f"void {name}({name}_a_row_t a_rows[{m}], {name}_res_row_t results[{m}]);"
    if interface == "stream" and not weights_in_core:
        return (f"void {name}(hls::stream<{name}_a_row_t> &a_stream, "
                f"hls::stream<{name}_b_col_t> &b_stream, "
                f"hls::stream<{name}_res_row_t> &res_stream);")
    return (f"void {name}(hls::stream<{name}_a_row_t> &a_stream, "
            f"hls::stream<{name}_res_row_t> &res_stream);")


def _call_and_fill(name, m, k, n, interface, weights_in_core):
    """The stimulus/fill + top call, leaving results in `results[M]` (res_row_t).

    Bias is the baked ROM (`{name}_bias_rom`, from `{name}_bias.h`) the top itself
    references -- never a call argument -- so the golden reads that same ROM
    (`bias_val`, matched by the caller to whatever values were baked) rather than
    building a local biases[] array to pass in.
    """
    if interface == "array" and not weights_in_core:
        return f"""    {name}_a_row_t a_rows[{m}];
    {name}_b_col_t b_cols[{n}];
    {name}_res_row_t results[{m}];
    for (int mm = 0; mm < {m}; mm++)
        for (int kk = 0; kk < {k}; kk++) a_rows[mm][kk] = a_val(mm, kk);
    for (int nn = 0; nn < {n}; nn++)
        for (int kk = 0; kk < {k}; kk++) b_cols[nn][kk] = b_val(kk, nn);
    {name}(a_rows, b_cols, results);
"""
    if interface == "array" and weights_in_core:
        return f"""    {name}_a_row_t a_rows[{m}];
    {name}_res_row_t results[{m}];
    for (int mm = 0; mm < {m}; mm++)
        for (int kk = 0; kk < {k}; kk++) a_rows[mm][kk] = a_val(mm, kk);
    {name}(a_rows, results);
"""
    if interface == "stream" and not weights_in_core:
        return f"""    hls::stream<{name}_a_row_t> a_stream("a");
    hls::stream<{name}_b_col_t> b_stream("b");
    hls::stream<{name}_res_row_t> res_stream("r");
    for (int mm = 0; mm < {m}; mm++) {{
        {name}_a_row_t a_row;
        for (int kk = 0; kk < {k}; kk++) a_row[kk] = a_val(mm, kk);
        a_stream.write(a_row);
    }}
    for (int nn = 0; nn < {n}; nn++) {{
        {name}_b_col_t b_col;
        for (int kk = 0; kk < {k}; kk++) b_col[kk] = b_val(kk, nn);
        b_stream.write(b_col);
    }}
    {name}(a_stream, b_stream, res_stream);
    {name}_res_row_t results[{m}];
    for (int mm = 0; mm < {m}; mm++) results[mm] = res_stream.read();
"""
    # stream + const_weights
    return f"""    hls::stream<{name}_a_row_t> a_stream("a");
    hls::stream<{name}_res_row_t> res_stream("r");
    for (int mm = 0; mm < {m}; mm++) {{
        {name}_a_row_t a_row;
        for (int kk = 0; kk < {k}; kk++) a_row[kk] = a_val(mm, kk);
        a_stream.write(a_row);
    }}
    {name}(a_stream, res_stream);
    {name}_res_row_t results[{m}];
    for (int mm = 0; mm < {m}; mm++) results[mm] = res_stream.read();
"""


def tb_cpp(name, m, k, n, interface="array", weights_in_core=False):
    inc_w = f'#include "{name}_weights.h"\n' if weights_in_core else ""
    # golden's B: const_weights reads the baked ROM; weighted uses b_val (same as fill).
    b_ref = (f"(double){name}_weight_cols_rom[nn][kk]"
             if weights_in_core else "b_val(kk, nn)")
    wl_label = " const_weights" if weights_in_core else ""
    return f"""#include <cstdio>
#include <cmath>
#include <hls_stream.h>
#include "{name}_config.h"
#include "{name}_gemm_ip.h"
#include "{name}_bias.h"
{inc_w}
{_prototype(name, m, k, n, interface, weights_in_core)}

static double a_val(int mm, int kk) {{ return (double)(((mm + kk) % 3) - 1); }}
static double b_val(int kk, int nn) {{ return (double)(((kk + 2 * nn) % 3) - 1); }}
// Golden reads the same baked bias ROM the top references (never a call argument).
static double bias_val(int nn) {{ return (double){name}_bias_rom[nn]; }}

int main() {{
{_call_and_fill(name, m, k, n, interface, weights_in_core)}
    int errors = 0;
    for (int mm = 0; mm < {m}; mm++) {{
        for (int nn = 0; nn < {n}; nn++) {{
            double golden = 0.0;
            for (int kk = 0; kk < {k}; kk++) golden += a_val(mm, kk) * ({b_ref});
            golden += bias_val(nn);
            double got = (double)results[mm][nn];
            if (std::fabs(got - golden) > 0.5) {{
                if (errors < 20)
                    std::printf("MISMATCH [%d][%d] got=%f exp=%f\\n", mm, nn, got, golden);
                errors++;
            }}
        }}
    }}
    if (errors == 0)
        std::printf("GENERIC CSIM PASS ({name} {m}x{k}x{n} {interface}{wl_label})\\n");
    else
        std::printf("GENERIC CSIM FAIL errors=%d\\n", errors);
    return errors ? 1 : 0;
}}
"""
