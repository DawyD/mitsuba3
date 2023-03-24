#include <mutex>
#include <tuple>

#include <drjit/morton.h>
#include <mitsuba/core/fwd.h>
#include <mitsuba/core/profiler.h>
#include <mitsuba/core/progress.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/timer.h>
#include <mitsuba/core/util.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/sampler.h>
#include <mitsuba/render/sensor.h>
#include <mitsuba/render/spiral.h>
#include <nanothread/nanothread.h>

#include <mitsuba/core/ray.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/records.h>


NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
----------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
     corresponds to :math:`\infty`). A value of 1 will only render directly
     visible light sources. 2 will lead to single-bounce (direct-only)
     illumination, and so on. (Default: -1)

 * - rr_depth
   - |int|
   - Specifies the path depth, at which the implementation will begin to use
     the *russian roulette* path termination criterion. For example, if set to
     1, then path generation many randomly cease after encountering directly
     visible surfaces. (Default: 5)

 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the intersection,
and the ray-casting step repeats over and over again (until one of several
stopping criteria applies).

.. image:: ../../resources/data/docs/images/integrator/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where a path tracer can be
expected to produce reasonable results: this is the case when the emitters are
easily "accessible" by the contents of the scene. For instance, an interior
scene that is lit by an area light will be considerably harder to render when
this area light is inside a glass enclosure (which effectively counts as an
occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. note:: This integrator does not handle participating media

.. tabs::
    .. code-tab::  xml
        :name: path-integrator

        <integrator type="path">
            <integer name="max_depth" value="8"/>
        </integrator>

    .. code-tab:: python

        'type': 'path',
        'max_depth': 8

 */

template <typename Float, typename Spectrum>
class PsPathIntegrator : public Integrator<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Integrator, should_stop, aov_names,
                   m_stop, m_timeout, m_render_timer, m_hide_emitters)
    MI_IMPORT_TYPES(Scene, Sensor, Film, ImageBlock, Medium, Sampler, Emitter, EmitterPtr, BSDF, BSDFPtr)

    PsPathIntegrator(const Properties &props) : Base(props) {

        m_block_size = props.get<uint32_t>("block_size", 0);

        // If a block size is specified, ensure that it is a power of two
        uint32_t block_size = math::round_to_power_of_two(m_block_size);
        if (m_block_size > 0 && block_size != m_block_size) {
            Log(Warn, "Setting block size from %i to next higher power of two: %i", m_block_size,
                block_size);
            m_block_size = block_size;
        }

        m_samples_per_pass = props.get<uint32_t>("samples_per_pass", (uint32_t) -1);
        if (m_samples_per_pass != (uint32_t) -1) {
            Log(Warn, "The 'samples_per_pass' is deprecated, as a poor choice of "
                      "this parameter can have a detrimental effect on performance. "
                      "Please leave it undefined; Mitsuba will then automatically "
                      "choose the necessary number of passes.");
        }

        int max_depth = props.get<int>("max_depth", -1);
        if (max_depth < 0 && max_depth != -1)
            Throw("\"max_depth\" must be set to -1 (infinite) or a value >= 0");

        m_max_depth = (uint32_t) max_depth; // This maps -1 to 2^32-1 bounces

        // Depth to begin using russian roulette
        int rr_depth = props.get<int>("rr_depth", 5);
        if (rr_depth <= 0)
            Throw("\"rr_depth\" must be set to a value greater than zero!");

        m_rr_depth = (uint32_t) rr_depth;
    }
    
    ~PsPathIntegrator() { }

    void render_sample(const Scene *scene,
                       const Sensor *sensor,
                       Sampler *sampler,
                       ImageBlock *block,
                       Float *aovs,
                       const Vector2f &pos,
                       ScalarFloat diff_scale_factor,
                       Mask active = true) {

        const Film *film = sensor->film();
        const bool has_alpha = has_flag(film->flags(), FilmFlags::Alpha);
        const bool box_filter = film->rfilter()->is_box_filter();

        ScalarVector2f scale = 1.f / ScalarVector2f(film->crop_size()),
                       offset = -ScalarVector2f(film->crop_offset()) * scale;

        Vector2f sample_pos   = pos + sampler->next_2d(active),
                 adjusted_pos = dr::fmadd(sample_pos, scale, offset);

        Point2f aperture_sample(.5f);
        if (sensor->needs_aperture_sample())
            aperture_sample = sampler->next_2d(active);

        Float time = sensor->shutter_open();
        if (sensor->shutter_open_time() > 0.f)
            time += sampler->next_1d(active) * sensor->shutter_open_time();

        Float wavelength_sample = 0.f;
        if constexpr (is_spectral_v<Spectrum>)
            wavelength_sample = sampler->next_1d(active);

        auto [ray, ray_weight] = sensor->sample_ray_differential(
            time, wavelength_sample, adjusted_pos, aperture_sample);

        if (ray.has_differentials)
            ray.scale_differential(diff_scale_factor);

        const Medium *medium = sensor->medium();


        auto [spec, valid] = sample(scene, sampler, ray, medium, nullptr, active);

        UnpolarizedSpectrum spec_u = unpolarized_spectrum(ray_weight * spec);

        if (unlikely(has_flag(film->flags(), FilmFlags::Special))) {
            film->prepare_sample(spec_u, ray.wavelengths, aovs,
                                 /*weight*/ 1.f,
                                 /*alpha */ dr::select(valid, Float(1.f), Float(0.f)),
                                 valid);
        } else {
            if constexpr (is_monochromatic_v<Spectrum> && nr_channels_v<Spectrum> != 1) {
                for (size_t i = 0; i < nr_channels_v<Spectrum>; ++i)
                    aovs[i] = spec_u[i];
            }
            else {
                Color3f rgb;
                if constexpr (is_spectral_v<Spectrum>)
                    rgb = spectrum_to_srgb(spec_u, ray.wavelengths, active);
                else if constexpr (is_monochromatic_v<Spectrum>)
                    rgb = spec_u.x();
                else
                    rgb = spec_u;

                aovs[0] = rgb.x();
                aovs[1] = rgb.y();
                aovs[2] = rgb.z();
            }

            if (likely(has_alpha)) {
                aovs[nr_channels_v<Spectrum>] = dr::select(valid, Float(1.f), Float(0.f));
                aovs[nr_channels_v<Spectrum>+1] = 1.f;
            } else {
                aovs[nr_channels_v<Spectrum>] = 1.f;
            }
        }

        // With box filter, ignore random offset to prevent numerical instabilities
        block->put(box_filter ? pos : sample_pos, aovs, active);
    }

    void render_block(const Scene *scene,
                      const Sensor *sensor,
                      Sampler *sampler,
                      ImageBlock *block,
                      Float *aovs,
                      uint32_t sample_count,
                      uint32_t seed,
                      uint32_t block_id,
                      uint32_t block_size) {

        if constexpr (!dr::is_array_v<Float>) {
            uint32_t pixel_count = block_size * block_size;

            // Avoid overlaps in RNG seeding RNG when a seed is manually specified
            seed += block_id * pixel_count;

            // Scale down ray differentials when tracing multiple rays per pixel
            Float diff_scale_factor = dr::rsqrt((Float) sample_count);

            // Clear block (it's being reused)
            block->clear();

            for (uint32_t i = 0; i < pixel_count && !should_stop(); ++i) {
                sampler->seed(seed + i);

                Point2u pos = dr::morton_decode<Point2u>(i);
                if (dr::any(pos >= block->size()))
                    continue;

                Point2f pos_f = Point2f(Point2i(pos) + block->offset());
                for (uint32_t j = 0; j < sample_count && !should_stop(); ++j) {
                    render_sample(scene, sensor, sampler, block, aovs, pos_f,
                                  diff_scale_factor);
                    sampler->advance();
                }
            }
        } else {
            DRJIT_MARK_USED(scene);
            DRJIT_MARK_USED(sensor);
            DRJIT_MARK_USED(sampler);
            DRJIT_MARK_USED(block);
            DRJIT_MARK_USED(aovs);
            DRJIT_MARK_USED(sample_count);
            DRJIT_MARK_USED(seed);
            DRJIT_MARK_USED(block_id);
            DRJIT_MARK_USED(block_size);
            Throw("Not implemented for JIT arrays.");
        }
    }

    TensorXf render(Scene *scene,
                    Sensor *sensor,
                    uint32_t seed = 0,
                    uint32_t spp = 0,
                    bool develop = true,
                    bool evaluate = true) override {
        m_stop = false;

        // Render on a larger film if the 'high quality edges' feature is enabled
        Film *film = sensor->film();
        ScalarVector2u film_size = film->crop_size();
        if (film->sample_border())
            film_size += 2 * film->rfilter()->border_size();

        // Potentially adjust the number of samples per pixel if spp != 0
        Sampler *sampler = sensor->sampler();
        if (spp)
            sampler->set_sample_count(spp);
        spp = sampler->sample_count();

        uint32_t spp_per_pass = (m_samples_per_pass == (uint32_t) -1)
                                    ? spp
                                    : std::min(m_samples_per_pass, spp);

        if ((spp % spp_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of spp_per_pass (%d).",
                  spp, spp_per_pass);

        uint32_t n_passes = spp / spp_per_pass;

        // Determine output channels and prepare the film with this information
        size_t n_channels = film->prepare(aov_names());

        // Start the render timer (used for timeouts & log messages)
        m_render_timer.reset();

        TensorXf result;


        if constexpr (!dr::is_jit_v<Float>) {
            if (film_size.x() == 1 && film_size.y() == 1)
            {
                ref<Sampler> sampler = sensor->sampler()->fork();

                ref<ImageBlock> block = film->create_block(
                    ScalarVector2u(1) /* size */,
                    false /* normalize */, true /* border */);

                std::unique_ptr<Float[]> aovs(new Float[n_channels]);

                block->set_size(film_size);
                block->set_offset(film->crop_offset());

                render_block(scene, sensor, sampler, block,
                             aovs.get(), spp_per_pass, seed,
                             0 /* block id */, 1 /* block size */);

                film->put_block(block);
            }
            else {
                // Render on the CPU using a spiral pattern
                uint32_t n_threads = (uint32_t) Thread::thread_count();

                Log(Info,
                    "Starting render job (%ux%u, %u sample%s,%s %u thread%s)",
                    film_size.x(), film_size.y(), spp, spp == 1 ? "" : "s",
                    n_passes > 1 ? tfm::format(" %u passes,", n_passes) : "",
                    n_threads, n_threads == 1 ? "" : "s");

                if (m_timeout > 0.f)
                    Log(Info, "Timeout specified: %.2f seconds.", m_timeout);

                // If no block size was specified, find size that is good for parallelization
                uint32_t block_size = m_block_size;
                if (block_size == 0) {
                    block_size = MI_BLOCK_SIZE; // 32x32
                    while (true) {
                        // Ensure that there is a block for every thread
                        if (block_size == 1 ||
                            dr::prod((film_size + block_size - 1) /
                                     block_size) >= n_threads)
                            break;
                        block_size /= 2;
                    }
                }

                Spiral spiral(film_size, film->crop_offset(), block_size,
                              n_passes);

                std::mutex mutex;
                ref<ProgressReporter> progress;
                Logger *logger = mitsuba::Thread::thread()->logger();
                if (logger && Info >= logger->log_level())
                    progress = new ProgressReporter("Rendering");

                // Total number of blocks to be handled, including multiple passes.
                uint32_t total_blocks = spiral.block_count() * n_passes,
                         blocks_done  = 0;

                // Grain size for parallelization
                uint32_t grain_size =
                    std::max(total_blocks / (4 * n_threads), 1u);

                // Avoid overlaps in RNG seeding RNG when a seed is manually specified
                seed *= dr::prod(film_size);

                ThreadEnvironment env;
                dr::parallel_for(
                    dr::blocked_range<uint32_t>(0, total_blocks, grain_size),
                    [&](const dr::blocked_range<uint32_t> &range) {
                        ScopedSetThreadEnvironment set_env(env);
                        // Fork a non-overlapping sampler for the current worker
                        ref<Sampler> sampler = sensor->sampler()->fork();

                        ref<ImageBlock> block = film->create_block(
                            ScalarVector2u(block_size) /* size */,
                            false /* normalize */, true /* border */);

                        std::unique_ptr<Float[]> aovs(new Float[n_channels]);

                        // Render up to 'grain_size' image blocks
                        for (uint32_t i = range.begin();
                             i != range.end() && !should_stop(); ++i) {
                            auto [offset, size, block_id] = spiral.next_block();
                            Assert(dr::prod(size) != 0);

                            if (film->sample_border())
                                offset -= film->rfilter()->border_size();

                            block->set_size(size);
                            block->set_offset(offset);

                            render_block(scene, sensor, sampler, block,
                                         aovs.get(), spp_per_pass, seed,
                                         block_id, block_size);

                            film->put_block(block);

                            /* Critical section: update progress bar */
                            if (progress) {
                                std::lock_guard<std::mutex> lock(mutex);
                                blocks_done++;
                                progress->update(blocks_done /
                                                 (float) total_blocks);
                            }
                        }
                    });
            }

            if (develop)
                result = film->develop();
        } else {

            size_t wavefront_size = (size_t) film_size.x() *
                                    (size_t) film_size.y() * (size_t) spp_per_pass,
                   wavefront_size_limit = 0xffffffffu;

            if (wavefront_size > wavefront_size_limit) {
                spp_per_pass /=
                    (uint32_t)((wavefront_size + wavefront_size_limit - 1) /
                                wavefront_size_limit);
                n_passes       = spp / spp_per_pass;
                wavefront_size = (size_t) film_size.x() * (size_t) film_size.y() *
                                 (size_t) spp_per_pass;

                Log(Warn,
                    "The requested rendering task involves %zu Monte Carlo "
                    "samples, which exceeds the upper limit of 2^32 = 4294967296 "
                    "for this variant. Mitsuba will instead split the rendering "
                    "task into %zu smaller passes to avoid exceeding the limits.",
                    wavefront_size, n_passes);
            }

            dr::sync_thread(); // Separate from scene initialization (for timings)

            Log(Info, "Starting render job (%ux%u, %u sample%s%s)",
                film_size.x(), film_size.y(), spp, spp == 1 ? "" : "s",
                n_passes > 1 ? tfm::format(", %u passes", n_passes) : "");

            if (n_passes > 1 && !evaluate) {
                Log(Warn, "render(): forcing 'evaluate=true' since multi-pass "
                          "rendering was requested.");
                evaluate = true;
            }

            // Inform the sampler about the passes (needed in vectorized modes)
            sampler->set_samples_per_wavefront(spp_per_pass);

            // Seed the underlying random number generators, if applicable
            sampler->seed(seed, (uint32_t) wavefront_size);

            // Allocate a large image block that will receive the entire rendering
            ref<ImageBlock> block = film->create_block();
            block->set_offset(film->crop_offset());

            // Only use the ImageBlock coalescing feature when rendering enough samples
            block->set_coalesce(block->coalesce() && spp_per_pass >= 4);

            // Compute discrete sample position
            UInt32 idx = dr::arange<UInt32>((uint32_t) wavefront_size);

            // Try to avoid a division by an unknown constant if we can help it
            uint32_t log_spp_per_pass = dr::log2i(spp_per_pass);
            if ((1u << log_spp_per_pass) == spp_per_pass)
                idx >>= dr::opaque<UInt32>(log_spp_per_pass);
            else
                idx /= dr::opaque<UInt32>(spp_per_pass);

            // Compute the position on the image plane
            Vector2u pos;
            pos.y() = idx / film_size[0];
            pos.x() = dr::fnmadd(film_size[0], pos.y(), idx);

            if (film->sample_border())
                pos -= film->rfilter()->border_size();

            pos += film->crop_offset();

            // Scale factor that will be applied to ray differentials
            ScalarFloat diff_scale_factor = dr::rsqrt((ScalarFloat) spp);

            Timer timer;
            std::unique_ptr<Float[]> aovs(new Float[n_channels]);

            // Potentially render multiple passes
            for (size_t i = 0; i < n_passes; i++) {
                render_sample(scene, sensor, sampler, block, aovs.get(), pos,
                              diff_scale_factor);

                if (n_passes > 1) {
                    sampler->advance(); // Will trigger a kernel launch of size 1
                    sampler->schedule_state();
                    dr::eval(block->tensor());
                }
            }

            film->put_block(block);

            if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                jit_flag(JitFlag::LoopRecord)) {
                Log(Info, "Computation graph recorded. (took %s)",
                    util::time_string((float) timer.reset(), true));
            }

            if (develop) {
                result = film->develop();
                dr::schedule(result);
            } else {
                film->schedule_storage();
            }

            if (evaluate) {
                dr::eval();

                if (n_passes == 1 && jit_flag(JitFlag::VCallRecord) &&
                    jit_flag(JitFlag::LoopRecord)) {
                    Log(Info, "Code generation finished. (took %s)",
                        util::time_string((float) timer.value(), true));

                    /* Separate computation graph recording from the actual
                       rendering time in single-pass mode */
                    m_render_timer.reset();
                }

                dr::sync_thread();
            }
        }

        if (!m_stop && (evaluate || !dr::is_jit_v<Float>))
            Log(Info, "Rendering finished. (took %s)",
                util::time_string((float) m_render_timer.value(), true));

        return result;
    }


    std::pair<Spectrum, typename SamplingIntegrator<Float, Spectrum>::Mask>
    sample(const Scene *scene,
           Sampler *sampler,
           const RayDifferential3f &ray_,
           const Medium *medium = nullptr,
           Float *aovs = nullptr,
           Mask active = true) const {
        MI_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        if (unlikely(m_max_depth == 0))
            return { 0.f, false };

        // --------------------- Configure loop state ----------------------

        Ray3f ray                     = Ray3f(ray_);
        Spectrum throughput           = 1.f;
        Spectrum result               = 0.f;
        Float eta                     = 1.f; // Tracks radiance scaling due to index of refraction changes
        UInt32 depth                  = 0;

        // If m_hide_emitters == false, the environment emitter will be visible
        Mask valid_ray                = !m_hide_emitters && dr::neq(scene->environment(), nullptr);

        // Variables caching information from the previous bounce
        Interaction3f prev_si         = dr::zeros<Interaction3f>();
        Float         prev_bsdf_pdf   = 1.f;
        Bool          prev_bsdf_delta = true;
        BSDFContext   bsdf_ctx;

        /* Set up a Dr.Jit loop. This optimizes away to a normal loop in scalar
           mode, and it generates either a a megakernel (default) or
           wavefront-style renderer in JIT variants. This can be controlled by
           passing the '-W' command line flag to the mitsuba binary or
           enabling/disabling the JitFlag.LoopRecord bit in Dr.Jit.

           The first argument identifies the loop by name, which is helpful for
           debugging. The subsequent list registers all variables that encode
           the loop state variables. This is crucial: omitting a variable may
           lead to undefined behavior. */
        dr::Loop<Bool> loop("Path Tracer", sampler, ray, throughput, result,
                            eta, depth, valid_ray, prev_si, prev_bsdf_pdf,
                            prev_bsdf_delta, active);

        /* Inform the loop about the maximum number of loop iterations.
           This accelerates wavefront-style rendering by avoiding costly
           synchronization points that check the 'active' flag. */
        loop.set_max_iterations(m_max_depth);

        while (loop(active)) {
            /* dr::Loop implicitly masks all code in the loop using the 'active'
               flag, so there is no need to pass it to every function */

            SurfaceInteraction3f si =
                scene->ray_intersect(ray,
                                     /* ray_flags = */ +RayFlags::All,
                                     /* coherent = */ dr::eq(depth, 0u));

            // ---------------------- Direct emission ----------------------

            /* dr::any_or() checks for active entries in the provided boolean
               array. JIT/Megakernel modes can't do this test efficiently as
               each Monte Carlo sample runs independently. In this case,
               dr::any_or<..>() returns the template argument (true) which means
               that the 'if' statement is always conservatively taken. */
            if (dr::any_or<true>(dr::neq(si.emitter(scene), nullptr))) {
                DirectionSample3f ds(scene, si, prev_si);
                Float em_pdf = 0.f;

                if (dr::any_or<true>(!prev_bsdf_delta))
                    em_pdf = scene->pdf_emitter_direction(prev_si, ds,
                                                          !prev_bsdf_delta);

                // Compute MIS weight for emitter sample from previous bounce
                Float mis_bsdf = mis_weight(prev_bsdf_pdf, em_pdf);

                // Accumulate, being careful with polarization (see spec_fma)
                result = spec_fma(
                    throughput,
                    ds.emitter->eval(si, prev_bsdf_pdf > 0.f) * mis_bsdf,
                    result);
            }

            // Continue tracing the path at this point?
            Bool active_next = (depth + 1 < m_max_depth) && si.is_valid();

            if (dr::none_or<false>(active_next))
                break; // early exit for scalar mode

            BSDFPtr bsdf = si.bsdf(ray);

            // ---------------------- Emitter sampling ----------------------

            // Perform emitter sampling?
            Mask active_em = active_next && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            DirectionSample3f ds = dr::zeros<DirectionSample3f>();
            Spectrum em_weight = dr::zeros<Spectrum>();
            Vector3f wo = dr::zeros<Vector3f>();

            if (dr::any_or<true>(active_em)) {
                for (auto const &scene_emitter : scene->emitters()) {
                    Mask active             = active_em;

                    std::tie(ds, em_weight) = scene_emitter->sample_direction(si, sampler->next_2d(), active);

                    active &= dr::neq(ds.pdf, 0.f);

                    // Mark occluded samples as invalid if requested by the user
                    if (dr::any_or<true>(active)) {
                        Mask occluded = scene->ray_test(si.spawn_ray_to(ds.p), active);
                        dr::masked(em_weight, occluded) = 0.f;
                        dr::masked(ds.pdf, occluded) = 0.f;
                    }

                    /* Given the detached emitter sample, recompute its contribution
                       with AD to enable light source optimization. */
                    if (dr::grad_enabled(si.p)) {
                        ds.d = dr::normalize(ds.p - si.p);
                        Spectrum em_val = scene->eval_emitter_direction(si, ds, active);
                        em_weight = dr::select(dr::neq(ds.pdf, 0), em_val / ds.pdf, 0);
                    }

                    // Query the BSDF for that emitter-sampled direction
                    wo = si.to_local(ds.d);

                    // Determine density of sampling that same direction using BSDF sampling
                    auto [bsdf_val, bsdf_pdf] = bsdf->eval_pdf(bsdf_ctx, si, wo, active);

                    if (dr::any_or<true>(active)) {
                        bsdf_val = si.to_world_mueller(bsdf_val, -wo, si.wi);

                        // Compute the MIS weight
                        Float mis_em = dr::select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));

                        // Accumulate, being careful with polarization (see spec_fma)
                        result[active] = spec_fma(throughput, bsdf_val * em_weight * mis_em, result);
                    }
                }
            }

            // ------ Evaluate BSDF * cos(theta) and sample direction -------

            Float sample_1   = sampler->next_1d();
            Point2f sample_2 = sampler->next_2d();

            auto [bsdf_sample, bsdf_weight] = bsdf->sample(bsdf_ctx, si, sample_1, sample_2);

            // ---------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            bsdf_weight = si.to_world_mueller(bsdf_weight, -bsdf_sample.wo, si.wi);

            // Intersect the BSDF ray against the scene geometry
            ray = si.spawn_ray(si.to_world(bsdf_sample.wo));

            /* When the path tracer is differentiated, we must be careful that
               the generated Monte Carlo samples are detached (i.e. don't track
               derivatives) to avoid bias resulting from the combination of moving
               samples and discontinuous visibility. We need to re-evaluate the
               BSDF differentiably with the detached sample in that case. */
            if (dr::grad_enabled(ray)) {
                ray = dr::detach<true>(ray);

                // Recompute 'wo' to propagate derivatives to cosine term
                Vector3f wo_2 = si.to_local(ray.d);
                auto [bsdf_val_2, bsdf_pdf_2] = bsdf->eval_pdf(bsdf_ctx, si, wo_2, active);
                bsdf_weight[bsdf_pdf_2 > 0.f] = bsdf_val_2 / dr::detach(bsdf_pdf_2);
            }

            // ------ Update loop variables based on current interaction ------

            throughput *= bsdf_weight;
            eta *= bsdf_sample.eta;
            valid_ray |= active && si.is_valid() &&
                         !has_flag(bsdf_sample.sampled_type, BSDFFlags::Null);

            // Information about the current vertex needed by the next iteration
            prev_si = si;
            prev_bsdf_pdf = bsdf_sample.pdf;
            prev_bsdf_delta = has_flag(bsdf_sample.sampled_type, BSDFFlags::Delta);

            // -------------------- Stopping criterion ---------------------

            dr::masked(depth, si.is_valid()) += 1;

            Float throughput_max = dr::max(unpolarized_spectrum(throughput));

            Float rr_prob = dr::minimum(throughput_max * dr::sqr(eta), .95f);
            Mask rr_active = depth >= m_rr_depth,
                 rr_continue = sampler->next_1d() < rr_prob;

            /* Differentiable variants of the renderer require the the russian
               roulette sampling weight to be detached to avoid bias. This is a
               no-op in non-differentiable variants. */
            throughput[rr_active] *= dr::rcp(dr::detach(rr_prob));

            active = active_next && (!rr_active || rr_continue) &&
                     dr::neq(throughput_max, 0.f);
        }

        return {
            /* spec  = */ dr::select(valid_ray, result, 0.f),
            /* valid = */ valid_ray
        };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PsPathIntegrator[\n"
            "  max_depth = %u,\n"
            "  rr_depth = %u\n"
            "]", m_max_depth, m_rr_depth);
    }

    /// Compute a multiple importance sampling weight using the power heuristic
    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        Float w = pdf_a / (pdf_a + pdf_b);
        return dr::detach<true>(dr::select(dr::isfinite(w), w, 0.f));
    }

    /**
     * \brief Perform a Mueller matrix multiplication in polarized modes, and a
     * fused multiply-add otherwise.
     */
    Spectrum spec_fma(const Spectrum &a, const Spectrum &b,
                      const Spectrum &c) const {
        if constexpr (is_polarized_v<Spectrum>)
            return a * b + c;
        else
            return dr::fmadd(a, b, c);
    }

    MI_DECLARE_CLASS()

    protected:
        uint32_t m_max_depth;
        uint32_t m_rr_depth;

        /// Size of (square) image blocks to render per core.
        uint32_t m_block_size;

        /**
         * \brief Number of samples to compute for each pass over the image blocks.
         *
         * Must be a multiple of the total sample count per pixel.
         * If set to (uint32_t) -1, all the work is done in a single pass (default).
         */
        uint32_t m_samples_per_pass;
};

MI_IMPLEMENT_CLASS_VARIANT(PsPathIntegrator, Integrator)
MI_EXPORT_PLUGIN(PsPathIntegrator, "Photometric Stereo Path Tracer integrator");
NAMESPACE_END(mitsuba)
