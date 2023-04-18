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

#define MI_ROUGH_TRANSMITTANCE_RES 64

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
        dr::set_attr(this, "flags", m_flags);

        // try to load the MERL BRDF data
        auto fs            = Thread::thread()->file_resolver();
        fs::path file_path = fs->resolve(props.string("filename"));
        m_name             = file_path.filename().string();

        load_MERL_file();

        if constexpr (dr::is_jit_v<Float>)
            Throw("Only scalar mode is supported for the MERL plugin at the moment!");


        m_eta = 1.0;
        m_specular_sampling_weight = props.get<ScalarFloat>("specular_sampling_weight",  .5f);
        m_sample_visible = true;
        parameters_changed();
    }

    std::pair<BSDFSample3f, Spectrum> sample(const BSDFContext &ctx,
                                             const SurfaceInteraction3f &si,
                                             Float sample1,
                                             const Point2f &sample2,
                                             Mask active) const override {

        MI_MASKED_FUNCTION(ProfilerPhase::BSDFSample, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi);
        active &= cos_theta_i > 0.f;

        BSDFSample3f bs = dr::zeros<BSDFSample3f>();
        Spectrum result(0.f);
        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return { bs, result };

        Float t_i = lerp_gather(m_external_transmittance, cos_theta_i,
                                MI_ROUGH_TRANSMITTANCE_RES, active);

        // Determine which component should be sampled
        Float prob_specular = (1.f - t_i) * m_specular_sampling_weight,
              prob_diffuse  = t_i * (1.f - m_specular_sampling_weight);

        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;
        else
            prob_specular = prob_specular / (prob_specular + prob_diffuse);
        prob_diffuse = 1.f - prob_specular;

        Mask sample_specular = active && (sample1 < prob_specular),
             sample_diffuse = active && !sample_specular;

        bs.eta = 1.f;

        if (dr::any_or<true>(sample_specular)) {
            MicrofacetDistribution distr(MicrofacetType::GGX, m_alpha_sample, m_sample_visible);
            Normal3f m = std::get<0>(distr.sample(si.wi, sample2));

            dr::masked(bs.wo, sample_specular) = reflect(si.wi, m);
            dr::masked(bs.sampled_component, sample_specular) = 0;
            dr::masked(bs.sampled_type, sample_specular) = +BSDFFlags::GlossyReflection;
        }

        if (dr::any_or<true>(sample_diffuse)) {
            dr::masked(bs.wo, sample_diffuse) = warp::square_to_cosine_hemisphere(sample2);
            dr::masked(bs.sampled_component, sample_diffuse) = 1;
            dr::masked(bs.sampled_type, sample_diffuse) = +BSDFFlags::DiffuseReflection;
        }

        bs.pdf = pdf(ctx, si, bs.wo, active);
        active &= bs.pdf > 0.f;
        result = eval(ctx, si, bs.wo, active);

        return { bs, (depolarizer<Spectrum>(result) / bs.pdf) & active };
    }

    Spectrum eval(const BSDFContext &ctx, const SurfaceInteraction3f &si,
                  const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse  = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

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


        // Compute specular and diffuse component
        if(!has_diffuse || !has_specular) {
            BSDFContext ctx2 = ctx;

            ctx2.type_mask |= (uint32_t)BSDFFlags::DiffuseReflection;
            ctx2.type_mask &= ~(uint32_t)BSDFFlags::GlossyReflection;


            //Compute specular component

            MicrofacetDistribution distr(MicrofacetType::GGX, m_alpha_sample, m_sample_visible);

            // Calculate the reflection half-vector
            Vector3f H = dr::normalize(wo + si.wi);

            // Evaluate the microfacet normal distribution
            Float D = distr.eval(H);

            // Fresnel term
            Float F = std::get<0>(fresnel(dr::dot(si.wi, H), Float(1.0)));  // m_eta set to 1.0

            // Smith's shadow-masking function
            Float G = distr.G(si.wi, wo, H);

            // Calculate the specular reflection component
            UnpolarizedSpectrum value_spec = F * D * G / (4.f * cos_theta_i);

            // compute specular and diffuse component

            UnpolarizedSpectrum value_diff = value - value_spec;

            // copy
            if(has_diffuse) value = dr::select(value_diff < 0.0, 0.0, value_diff);
            else value = dr::select(value_diff < 0.0, value, value_spec);
        }

        value *= cos_theta_o;

        return depolarizer<Spectrum>(value) & active;
    }

    Float lerp_gather(const DynamicBuffer<Float> &data, Float x, size_t size,
                      Mask active = true) const {
        using UInt32 = dr::uint32_array_t<Float>;
        x *= Float(size - 1);
        UInt32 index = dr::minimum(UInt32(x), uint32_t(size - 2));

        Float v0 = dr::gather<Float>(data, index, active),
              v1 = dr::gather<Float>(data, index + 1, active);

        return dr::lerp(v0, v1, x - Float(index));
    }

    Float pdf(const BSDFContext &ctx, const SurfaceInteraction3f &si,
              const Vector3f &wo, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::BSDFEvaluate, active);

        bool has_specular = ctx.is_enabled(BSDFFlags::GlossyReflection, 0),
             has_diffuse = ctx.is_enabled(BSDFFlags::DiffuseReflection, 1);

        Float cos_theta_i = Frame3f::cos_theta(si.wi),
              cos_theta_o = Frame3f::cos_theta(wo);

        active &= cos_theta_i > 0.f && cos_theta_o > 0.f;

        if (unlikely((!has_specular && !has_diffuse) || dr::none_or<false>(active)))
            return 0.f;

        Float t_i = lerp_gather(m_external_transmittance, cos_theta_i,
                                MI_ROUGH_TRANSMITTANCE_RES, active);

        // Determine which component should be sampled
        Float prob_specular = (1.f - t_i) * m_specular_sampling_weight,
              prob_diffuse  = t_i * (1.f - m_specular_sampling_weight);

        if (unlikely(has_specular != has_diffuse))
            prob_specular = has_specular ? 1.f : 0.f;
        else
            prob_specular = prob_specular / (prob_specular + prob_diffuse);
        prob_diffuse = 1.f - prob_specular;

        Vector3f H = dr::normalize(wo + si.wi);

        MicrofacetDistribution distr(MicrofacetType::GGX, m_alpha_sample, m_sample_visible);
        Float result = 0.f;
        if (m_sample_visible)
            result = distr.eval(H) * distr.smith_g1(si.wi, H) /
                     (4.f * cos_theta_i);
        else
            result = distr.pdf(si.wi, H) / (4.f * dr::dot(wo, H));
        result *= prob_specular;

        result += prob_diffuse * warp::square_to_cosine_hemisphere_pdf(wo);

        return result;
    }

    void traverse(TraversalCallback *callback) override {
        Base::traverse(callback);
        callback->put_parameter("name", m_name, +ParamFlags::NonDifferentiable);
        callback->put_parameter("alpha_sample", m_alpha_sample, ParamFlags::Differentiable | ParamFlags::Discontinuous);
        callback->put_parameter("specular_sampling_weight", m_specular_sampling_weight, +ParamFlags::NonDifferentiable);
        // callback->put_parameter("data", m_data, +ParamFlags::NonDifferentiable);
    }

    void parameters_changed(const std::vector<std::string> &keys = {}) override {
        if (keys.empty() || string::contains(keys, "name"))
            load_MERL_file();

        // Precompute rough reflectance (vectorized)
        if (keys.empty() || string::contains(keys, "alpha_sample")) {
            using FloatX = DynamicBuffer<ScalarFloat>;
            using Vector3fX = Vector<FloatX, 3>;
            ScalarFloat eta = dr::slice(m_eta), alpha = dr::slice(m_alpha_sample);
            using FloatP = dr::Packet<dr::scalar_t<Float>>;
            mitsuba::MicrofacetDistribution<FloatP, Spectrum> distr(MicrofacetType::GGX, alpha);
            FloatX mu = dr::maximum(1e-6f, dr::linspace<FloatX>(0, 1, MI_ROUGH_TRANSMITTANCE_RES));
            FloatX zero = dr::zeros<FloatX>(MI_ROUGH_TRANSMITTANCE_RES);
            Vector3fX wi = Vector3fX(dr::sqrt(1 - mu * mu), zero, mu);
            auto external_transmittance = eval_transmittance(distr, wi, eta);
            m_external_transmittance = dr::load<DynamicBuffer<Float>>(
                external_transmittance.data(),
                dr::width(external_transmittance));
        }
        dr::make_opaque(m_eta, m_alpha_sample, m_specular_sampling_weight);

        Base::parameters_changed(keys);

    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "MERL[" << std::endl
            << "  name = " << m_name << std::endl
            << "  alpha_sample = " << m_alpha_sample << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
private:
    std::string m_name;
    std::vector<double> m_data;
    ScalarFloat m_alpha_sample;
    ScalarFloat m_specular_sampling_weight;

    Float m_eta;
    DynamicBuffer<Float> m_external_transmittance;
    bool m_sample_visible;

    void load_MERL_file(){
        if(!read_brdf(m_name.c_str(), m_data))
            Log(Error, "Unable to find \"%s\".", m_name.c_str());
    }
};

MI_IMPLEMENT_CLASS_VARIANT(MERL, BSDF)
MI_EXPORT_PLUGIN(MERL, "MERL material")
NAMESPACE_END(mitsuba)
