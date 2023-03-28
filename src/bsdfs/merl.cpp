#include "brdf_merl.h"
#include <mitsuba/core/distr_2d.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/warp.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/microfacet.h>
#include <mitsuba/render/texture.h>
#include <drjit/struct.h>

NAMESPACE_BEGIN(mitsuba)

/* Set the weight for cosine hemisphere sampling in relation to GGX sampling.
   Set to 1.0 in order to fully fall back to cosine sampling. */
#define COSINE_HEMISPHERE_PDF_WEIGHT 0.1f

/**!

.. _bsdf-merl:

MERL material (:monosp:`merl`)
----------------------------------------------------------

.. pluginparameters::

 * - filename
   - |string|
   - Filename of the material MERL data file to be loaded

 * - alpha_sample
   - |float|
   - Specifies which roughness value should be used for the internal Microfacet
     importance sampling routine. (Default: 0.1)

Internally, a sampling routine from the GGX Microfacet model is used in order to
importance sampling outgoing directions. The used GGX roughness value is exposed
here as a user parameter `alpha_sample` and should be set according to the
approximate roughness of the material to be rendered. Note that any value here
will result in a correct rendering but the level of noise can vary significantly.

*/

template <typename Float, typename Spectrum>
class MERL final : public BSDF<Float, Spectrum> {
public:
    MI_IMPORT_BASE(BSDF, m_flags, m_components)
    MI_IMPORT_TYPES(Texture, MicrofacetDistribution)

    MERL(const Properties &props) : Base(props) {
        m_alpha_sample = props.get<ScalarFloat>("alpha_sample", 0.1f);

        m_components.push_back(BSDFFlags::GlossyReflection | BSDFFlags::FrontSide);
        m_components.push_back(BSDFFlags::DiffuseReflection | BSDFFlags::FrontSide);
        m_flags =  m_components[0] | m_components[1];

        // try to load the MERL BRDF data
        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name             = file_path.filename().string();

        load_MERL_file();

        if constexpr (dr::is_jit_v<Float>)
            Throw("Only scalar mode is supported for the MERL plugin at the moment!");
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {

        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs;
        if (unlikely(dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection)))
            return { bs, 0.f };

        MicrofacetDistribution distr(MicrofacetType::GGX,
                                     m_alpha_sample, m_alpha_sample, true);

        Float lobe_pdf_diffuse = COSINE_HEMISPHERE_PDF_WEIGHT;
        Mask sample_diffuse    = active && sample1 < lobe_pdf_diffuse,
             sample_microfacet = active && !sample_diffuse;

        Vector3f wo_diffuse    = warp::square_to_cosine_hemisphere(sample2);
        auto [m, unused] = distr.sample(si.wi, sample2);
        Vector3f wo_microfacet = reflect(si.wi, m);

        bs.wo[sample_diffuse]    = wo_diffuse;
        bs.wo[sample_microfacet] = wo_microfacet;

        bs.pdf = pdf(ctx, si, bs.wo, active);

        bs.sampled_component = 0;
        bs.sampled_type = +BSDFFlags::GlossyReflection;
        bs.eta = 1.f;

        Spectrum value = eval(ctx, si, bs.wo, active);
        return { bs, dr::select(active && bs.pdf > 0, value / bs.pdf, 0.f) };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely(!has_diffuse || dr::none_or<false>(active)))
            return 0.f;

        //TODO: compute specular and diffuse component

        double r,g,b;

        Float twi = dr::acos(cos_theta_i);
        Float two = dr::acos(cos_theta_o);

        Float pwi = dr::atan2(si.wi.y(), si.wi.x());
        Float pwo = dr::atan2(wo.y(), wo.x());

        if constexpr (!dr::is_jit_v<Float>)
            lookup_brdf_val(m_data, (double) twi, (double) pwi, (double) two, (double) pwo, r, g, b);
        else
            Throw("Only scalar mode is supported for the MERL plugin at the moment!");

        UnpolarizedSpectrum value;
        if constexpr (is_monochromatic_v<Spectrum>)
            value = luminance(Color3f(r, g, b));
        else if constexpr (is_rgb_v<Spectrum>)
            value = Color3f(r, g, b);
        else 
            Throw("Only the monochrome and RGB modes are supported!");

        return depolarizer<Spectrum>(value) & active;
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        if (unlikely(dr::none_or<false>(active) || !ctx.is_enabled(BSDFFlags::GlossyReflection)))
            return 0.f;

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        MicrofacetDistribution distr(MicrofacetType::GGX,
                                     m_alpha_sample, m_alpha_sample, true);

        Vector3f H = dr::normalize(wo + si.wi);

        Float pdf_diffuse = warp::square_to_cosine_hemisphere_pdf(wo);
        Float pdf_microfacet = distr.pdf(si.wi, H) / (4.f * dr::dot(wo, H));

        Float pdf = 0.f;
        pdf += pdf_diffuse * COSINE_HEMISPHERE_PDF_WEIGHT;
        pdf += pdf_microfacet * (1.f - COSINE_HEMISPHERE_PDF_WEIGHT);

        return dr::select(cos_theta_i > 0.f && cos_theta_o > 0.f, pdf, 0.f);
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("name", m_name, +ParamFlags::NonDifferentiable);
        callback->put_parameter("alpha_sample", m_alpha_sample, ParamFlags::Differentiable | ParamFlags::Discontinuous);
        // callback->put_parameter("data", m_data, +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys) override {
        if (keys.empty() || string::contains(keys, "name"))
            load_MERL_file();
        Base::parameters_changed(keys);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MeasuredPolarized[" << std::endl
            << "  name = " << m_name << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    std::string m_name;
    std::vector<double> m_data;
    ScalarFloat m_alpha_sample;

    void load_MERL_file(){
        if(!read_brdf(m_name.c_str(), m_data))
            Log(Error, "Unable to find \"%s\".", m_name.c_str());
    }
};

MI_IMPLEMENT_CLASS_VARIANT(MERL, BSDF)
MI_EXPORT_PLUGIN(MERL, "MERL material")
NAMESPACE_END(mitsuba)
