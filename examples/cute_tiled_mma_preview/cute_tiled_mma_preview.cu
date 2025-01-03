#include <cassert>
#include <fstream>
#include <iomanip>
#include <iostream>

#include <cute/layout.hpp>
#include <cute/swizzle.hpp>
#include <cute/tensor.hpp>

#include <cutlass/util/command_line.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

class Swizzle
{
public:
    Swizzle(int num_bits, int num_base, int num_shft)
        : m_num_bits(num_bits), m_num_base(num_base), m_num_shft(num_shft)
    {
        assert(m_num_bits >= 0 && "BBits must be positive.");
        assert(m_num_base >= 0 && "MBase must be positive.");
        assert(std::abs(m_num_shft) >= m_num_bits &&
               "abs(SShift) must be more than BBits.");
    }

    template <class Offset>
    auto apply(Offset offset) const noexcept
    {
        return offset ^ shiftr(offset & m_yyy_msk); // ZZZ ^= YYY
    }

    template <class Offset>
    auto operator()(Offset offset) const noexcept
    {
        return apply(offset);
    }

private:
    template <class Offset>
    auto shiftr(Offset offset) const noexcept
    {
        return m_msk_sft >= 0 ? offset >> m_msk_sft : offset << -m_msk_sft;
    }

    int m_num_bits;
    int m_num_base;
    int m_num_shft;

    int m_bit_msk = (1 << m_num_bits) - 1;
    int m_yyy_msk = m_bit_msk << (m_num_base + std::max(0, m_num_shft));
    int m_zzz_msk = m_bit_msk << (m_num_base - std::min(0, m_num_shft));
    int m_msk_sft = m_num_shft;
};

template <class LayoutA, class TikzColorFn = cute::TikzColor_BWx8>
void print_layout_shared_memory_bank_id_latex(
    LayoutA const& layout_a, // (m,n) -> idx
    Swizzle const& swizzle, size_t element_size, std::ofstream& file,
    TikzColorFn color = {}) // lambda(idx) -> tikz color string
{
    CUTE_STATIC_ASSERT_V(cute::rank(layout_a) <= cute::Int<2>{});
    auto layout = cute::append<2>(layout_a, cute::Layout<cute::_1, cute::_0>{});

    // Header
    file << "\\documentclass[convert]{standalone}\n"
         << "\\usepackage{tikz}\n\n"
         << "\\begin{document}\n"
         << "\\begin{tikzpicture}[x={(0cm,-1cm)},y={(1cm,0cm)},every "
            "node/.style={minimum size=1cm, outer sep=0pt}]\n\n";
    // Layout
    for (int i = 0; i < cute::size<0>(layout); ++i)
    {
        for (int j = 0; j < cute::size<1>(layout); ++j)
        {
            int idx = layout(i, j);
            int swizzled_idx = swizzle(idx);
            int bank_id = (swizzled_idx * element_size / 4) % 32;
            file << "\\node[fill=" << color(bank_id) << "] at (" << i << ","
                 << j << ") {" << bank_id << "};\n";
        }
    }
    // Grid
    file << "\\draw[color=black,thick,shift={(-0.5,-0.5)}] (0,0) grid ("
         << int(cute::size<0>(layout)) << "," << int(cute::size<1>(layout))
         << ");\n\n";
    // Labels
    for (int i = 0, j = -1; i < cute::size<0>(layout); ++i)
    {
        file << "\\node at (" << i << "," << j << ") {\\Large{\\texttt{" << i
             << "}}};\n";
    }
    for (int i = -1, j = 0; j < cute::size<1>(layout); ++j)
    {
        file << "\\node at (" << i << "," << j << ") {\\Large{\\texttt{" << j
             << "}}};\n";
    }
    // Footer
    file << "\\end{tikzpicture}\n"
         << "\\end{document}\n";
}

std::ostream& print_usage(std::ostream& out)
{
    out << "Print the shared memory bank ids for a given shared memory layout "
           "and swizzle configuration.\n"
        << "\n"
        << "Options:\n"
        << "\n"
        << "  --help                            If specified, displays this "
           "usage statement.\n\n"
        << "  --m=<int>                         Matrix on shared memory M "
           "dimension\n"
        << "  --n=<int>                         Matrix on shared memory N "
           "dimension\n"
        << "  --stride_m=<int>                  Matrix on shared memory M "
           "stride\n"
        << "  --stride_n=<int>                  Matrix on shared memory N "
           "stride\n"
        << "  --element_size=<int>              Element size in bytes\n"
        << "  --swizzle_num_mask_bits=<int>     Number of swizzle mask bits\n"
        << "  --swizzle_num_base=<int>          Number of swizzle base bits\n"
        << "  --swizzle_num_shift=<int>         Number of swizzle shift bits\n"
        << "  --latex_file_path=<string>        LaTeX file path\n";

    out << "\nExamples:\n\n"
        << "$ "
        << " --m=32 --n=64 --stride_m=64 --stride_n=1 --element_size=4 "
           "--swizzle_num_mask_bits=5 --swizzle_num_base=0 "
           "--swizzle_num_shift=6 "
           "--latex_file_path=shared_memory_bank_ids.tex\n";

    return out;
}

int main(int argc, const char** argv)
{
    // Configure data type.
    using TA = cute::half_t;
    using TB = cute::half_t;

    // Configure static "shared memory".
    // The "shared memory" is actually on host for preview purpose.
    // For tiled mma, the shared memory layout has to be static.
    constexpr int bM{128 * 2 / sizeof(TA)};
    constexpr int bN{128 * 2 / sizeof(TB)};
    constexpr int bK{32};
    auto const blk_M = cute::Int<bM>{};
    auto const blk_N = cute::Int<bN>{};
    auto const blk_K = cute::Int<bK>{};

    auto const smem_shape_A{cute::make_shape(blk_M, blk_K)};
    auto const smem_shape_B{cute::make_shape(blk_N, blk_K)};
    auto const smem_stride_A{
        cute::make_stride(cute::Int<1>{}, blk_M)}; // Column-major
    auto const smem_stride_B{
        cute::make_stride(cute::Int<1>{}, blk_N)}; // Column-major
    auto const smem_layout_A{
        cute::make_layout(smem_shape_A, smem_stride_A)}; // (blk_M, blk_K)
    auto const smem_layout_B{
        cute::make_layout(smem_shape_B, smem_stride_B)}; // (blk_N, blk_K)

    auto const size_a{blk_M * blk_K};
    auto const size_b{blk_N * blk_K};

    auto h_A = thrust::host_vector<TA>(size_a);
    auto h_B = thrust::host_vector<TB>(size_b);

    // Make tensor for smem_A and smem_B.
    auto smem_tensor_A{cute::make_tensor(h_A.data(), smem_layout_A)};
    auto smem_tensor_B{cute::make_tensor(h_B.data(), smem_layout_B)};

    std::cout << "smem_tensor_A" << std::endl;
    cute::print(smem_tensor_A);
    std::cout << std::endl;
    std::cout << "smem_tensor_B" << std::endl;
    cute::print(smem_tensor_B);
    std::cout << std::endl;

    // Configure tiled MMA.
    using MmaTraits = cute::MMA_Traits<cute::SM80_16x8x16_F16F16F16F16_TN>;
    using MmaAtomShape = MmaTraits::Shape_MNK;
    auto const mma_atom = cute::MMA_Atom<MmaTraits>{};
    auto const mma_atom_shape = MmaAtomShape{};
    // Repeating the mma atom along the M, N, and K dimensions.
    // This increases the number of threads to process the tiled MMA.
    constexpr int MMA_LAYOUT_M{2};
    constexpr int MMA_LAYOUT_N{2};
    constexpr int MMA_LAYOUT_K{1};
    auto mma_layout{cute::make_layout(
        cute::make_shape(cute::Int<MMA_LAYOUT_M>{}, cute::Int<MMA_LAYOUT_N>{},
                         cute::Int<MMA_LAYOUT_K>{}))};
    // Repeating the mma processing along the M, N, and K dimensions.
    // This does not increase the number of threads to process the tiled MMA.
    // But the number of registers required for processing the tiled MMA
    // increases.
    constexpr int NUM_MMA_TILE_M{1};
    constexpr int NUM_MMA_TILE_N{2};
    constexpr int NUM_MMA_TILE_K{1};
    constexpr int MMA_TILE_M{cute::get<0>(mma_atom_shape) * MMA_LAYOUT_M *
                             NUM_MMA_TILE_M};
    constexpr int MMA_TILE_N{cute::get<1>(mma_atom_shape) * MMA_LAYOUT_N *
                             NUM_MMA_TILE_N};
    constexpr int MMA_TILE_K{cute::get<2>(mma_atom_shape) * MMA_LAYOUT_K *
                             NUM_MMA_TILE_K};
    auto mma_tile{cute::make_tile(cute::Int<MMA_TILE_M>{},
                                  cute::Int<MMA_TILE_N>{},
                                  cute::Int<MMA_TILE_K>{})};
    auto tiled_mma{cute::make_tiled_mma(mma_atom, mma_layout, mma_tile)};

    constexpr auto NUM_THREADS{cute::size(tiled_mma)};
    CUTE_STATIC_ASSERT(NUM_THREADS ==
                       MMA_LAYOUT_M * MMA_LAYOUT_N * MMA_LAYOUT_K *
                           cute::size(decltype(mma_atom)::ThrID{}));

    std::cout << "tiled_mma" << std::endl;
    cute::print(tiled_mma);
    std::cout << std::endl;

    // ThrLayoutVMNK static asserts.
    CUTE_STATIC_ASSERT_V(cute::shape<0>(decltype(tiled_mma)::ThrLayoutVMNK{}) ==
                         cute::shape(decltype(mma_atom)::ThrID{}));
    CUTE_STATIC_ASSERT_V(cute::shape<1>(decltype(tiled_mma)::ThrLayoutVMNK{}) ==
                         cute::Int<MMA_LAYOUT_M>{});
    CUTE_STATIC_ASSERT_V(cute::shape<2>(decltype(tiled_mma)::ThrLayoutVMNK{}) ==
                         cute::Int<MMA_LAYOUT_N>{});
    CUTE_STATIC_ASSERT_V(cute::shape<3>(decltype(tiled_mma)::ThrLayoutVMNK{}) ==
                         cute::Int<MMA_LAYOUT_K>{});

    // PermutationMNK static asserts.
    CUTE_STATIC_ASSERT_V(tiled_mma.tile_size_mnk<0>() ==
                         cute::Int<MMA_TILE_M>{});
    CUTE_STATIC_ASSERT_V(tiled_mma.tile_size_mnk<1>() ==
                         cute::Int<MMA_TILE_N>{});
    CUTE_STATIC_ASSERT_V(tiled_mma.tile_size_mnk<2>() ==
                         cute::Int<MMA_TILE_K>{});

    // Partition via MMA.
    // set an arbitrary thread index.
    constexpr int THREAD_IDX{0};
    CUTE_STATIC_ASSERT(THREAD_IDX < NUM_THREADS);
    CUTE_STATIC_ASSERT(THREAD_IDX >= 0);

    auto thread_mma{tiled_mma.get_slice(THREAD_IDX)};
    // Register tensors used for MMA.
    auto thread_layout_C_register_tensor_A{
        thread_mma.partition_fragment_A(smem_tensor_A)}; // (MMA, MMA_M, MMA_K)
    auto thread_layout_C_register_tensor_B{
        thread_mma.partition_fragment_B(smem_tensor_B)}; // (MMA, MMA_N, MMA_K)

    CUTE_STATIC_ASSERT_V(
        cute::shape<1>(decltype(mma_atom)::LayoutA_TV{}) ==
        cute::shape<0>(thread_layout_C_register_tensor_A)); // MMA_A
    CUTE_STATIC_ASSERT_V(
        cute::shape<1>(decltype(mma_atom)::LayoutB_TV{}) ==
        cute::shape<0>(thread_layout_C_register_tensor_B)); // MMA_B

    // Use no tiled copy from shared memory to register.
    auto thread_layout_C_smem_tensor_A_no_tiled_copy{
        thread_mma.partition_A(smem_tensor_A)}; // (MMA, MMA_M, MMA_K)
    auto thread_layout_C_smem_tensor_B_no_tiled_copy{
        thread_mma.partition_B(smem_tensor_B)}; // (MMA, MMA_N, MMA_K)

    // thread_layout_C_smem_tensor_A_no_tiled_copy and
    // thread_layout_C_register_tensor_A shall have the same shape.
    CUTE_STATIC_ASSERT_V(
        cute::shape(thread_layout_C_smem_tensor_A_no_tiled_copy) ==
        cute::shape(thread_layout_C_register_tensor_A));

    std::cout << "thread_layout_C_register_tensor_A" << std::endl;
    cute::print(thread_layout_C_register_tensor_A);
    std::cout << std::endl;
    std::cout << "thread_layout_C_register_tensor_B" << std::endl;
    cute::print(thread_layout_C_register_tensor_B);
    std::cout << std::endl;

    std::cout << "thread_layout_C_smem_tensor_A_no_tiled_copy" << std::endl;
    cute::print(thread_layout_C_smem_tensor_A_no_tiled_copy);
    std::cout << std::endl;
    std::cout << "thread_layout_C_smem_tensor_B_no_tiled_copy" << std::endl;
    cute::print(thread_layout_C_smem_tensor_B_no_tiled_copy);
    std::cout << std::endl;

    // Use tiled copy from shared memory to register.
    auto copy_atom_A = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, TA>{};
    auto copy_atom_B = cute::Copy_Atom<cute::SM75_U16x8_LDSM_T, TB>{};

    auto smem_tiled_copy_A{cute::make_tiled_copy_A(copy_atom_A, tiled_mma)};
    auto smem_tiled_copy_B{cute::make_tiled_copy_B(copy_atom_B, tiled_mma)};

    CUTE_STATIC_ASSERT_V(
        cute::shape<0>(decltype(smem_tiled_copy_A)::Tiler_MN{}) ==
        tiled_mma.tile_size_mnk<0>()); // MMA_TILE_M
    CUTE_STATIC_ASSERT_V(
        cute::shape<1>(decltype(smem_tiled_copy_A)::Tiler_MN{}) ==
        tiled_mma.tile_size_mnk<2>()); // MMA_TILE_K
    CUTE_STATIC_ASSERT_V(
        cute::shape<0>(decltype(smem_tiled_copy_B)::Tiler_MN{}) ==
        tiled_mma.tile_size_mnk<1>()); // MMA_TILE_N
    CUTE_STATIC_ASSERT_V(
        cute::shape<1>(decltype(smem_tiled_copy_B)::Tiler_MN{}) ==
        tiled_mma.tile_size_mnk<2>()); // MMA_TILE_K

    auto smem_thread_copy_A{smem_tiled_copy_A.get_slice(THREAD_IDX)};
    auto smem_thread_copy_B{smem_tiled_copy_B.get_slice(THREAD_IDX)};

    auto thread_layout_C_smem_tensor_A_tiled_copy{
        smem_thread_copy_A.partition_S(smem_tensor_A)};
    auto thread_layout_C_smem_tensor_B_tiled_copy{
        smem_thread_copy_B.partition_S(smem_tensor_B)};

    auto thread_layout_C_register_tensor_A_copy_view{
        smem_thread_copy_A.retile_D(thread_layout_C_register_tensor_A)};
    auto thread_layout_C_register_tensor_B_copy_view{
        smem_thread_copy_B.retile_D(thread_layout_C_register_tensor_B)};

    CUTE_STATIC_ASSERT_V(
        cute::shape(thread_layout_C_smem_tensor_A_tiled_copy) ==
        cute::shape(thread_layout_C_register_tensor_A_copy_view));
    CUTE_STATIC_ASSERT_V(
        cute::shape(thread_layout_C_smem_tensor_B_tiled_copy) ==
        cute::shape(thread_layout_C_register_tensor_B_copy_view));

    std::cout << "copy_atom_A" << std::endl;
    cute::print(copy_atom_A);
    std::cout << std::endl;
    std::cout << "copy_atom_B" << std::endl;
    cute::print(copy_atom_B);
    std::cout << std::endl;

    std::cout << "smem_tiled_copy_A" << std::endl;
    cute::print(smem_tiled_copy_A);
    std::cout << std::endl;
    std::cout << "smem_tiled_copy_B" << std::endl;
    cute::print(smem_tiled_copy_B);
    std::cout << std::endl;

    std::cout << "thread_layout_C_smem_tensor_A_tiled_copy" << std::endl;
    cute::print(thread_layout_C_smem_tensor_A_tiled_copy);
    std::cout << std::endl;
    std::cout << "thread_layout_C_smem_tensor_B_tiled_copy" << std::endl;
    cute::print(thread_layout_C_smem_tensor_B_tiled_copy);
    std::cout << std::endl;

    std::cout << "thread_layout_C_register_tensor_A_copy_view" << std::endl;
    cute::print(thread_layout_C_register_tensor_A_copy_view);
    std::cout << std::endl;
    std::cout << "thread_layout_C_register_tensor_B_copy_view" << std::endl;
    cute::print(thread_layout_C_register_tensor_B_copy_view);
    std::cout << std::endl;

    cute::print_latex(tiled_mma);

    return 0;
}
