#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/texture.h>
#include <mitsuba/render/srgb.h>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _spectrum-avx:

sRGB spectrum (:monosp:`avx`)
------------------------------

.. pluginparameters::
 :extra-rows: 1

 * - color
   - :paramtype:`color`
   - The corresponding AVX color value.

 * - value
   - :paramtype:`color`
   - Spectral upsampling model coefficients of the avx color value.
   - |exposed|, |differentiable|

 The plugin is only intended to be used in the AVX mode
In AVX render modes, this spectrum represents a constant AVX value.

.. tabs::
    .. code-tab:: xml
        :name: avx

        <spectrum type="avx">
            <rgb name="color" value="10, 20, 250, 10, 233, 12, 12, 98"/>
        </spectrum>

    .. code-tab:: python

        'type': 'srgb',
        'color': [10, 20, 250, 10, 233, 12, 12, 98]

 */

template <typename Float, typename Spectrum>
class AVXReflectanceSpectrum final : public Texture<Float, Spectrum> {
public:
    MI_IMPORT_TYPES(Texture)

    AVXReflectanceSpectrum(const Properties &props) : Texture(props) {
        ScalarColor8f color = props.get<ScalarColor8f>("color8");

        if (dr::any(color < 0 || color > 1) && !props.get<bool>("unbounded", false))
            Throw("Invalid RGB reflectance value %s, must be in the range [0, 1]!", color);

        if constexpr (is_monochromatic_v<Spectrum> && nr_channels_v<Spectrum> == 8)
            m_value = color;
        else
            Throw("AVXReflectance spectrum can be only used in the avx mode");

        dr::make_opaque(m_value);
    }

    void traverse(TraversalCallback *callback) override {
        callback->put_parameter("value", m_value, +ParamFlags::Differentiable);
    }

    void parameters_changed(const std::vector<std::string> &/*keys*/ = {}) override {
        dr::make_opaque(m_value);
    }

    UnpolarizedSpectrum eval(const SurfaceInteraction3f &si, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);

        if constexpr (is_monochromatic_v<Spectrum> && nr_channels_v<Spectrum> == 8)
            return m_value;
        else
            Throw("AVXReflectance spectrum can be only used in the avx mode");
    }

    Float eval_1(const SurfaceInteraction3f & /*it*/, Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureEvaluate, active);
        return mean();
    }

    std::pair<Wavelength, UnpolarizedSpectrum>
    sample_spectrum(const SurfaceInteraction3f &_si,
                    const Wavelength &sample,
                    Mask active) const override {
        MI_MASKED_FUNCTION(ProfilerPhase::TextureSample, active);

        if constexpr (is_spectral_v<Spectrum>) {
            // TODO: better sampling strategy
            SurfaceInteraction3f si(_si);
            si.wavelengths = MI_CIE_MIN + (MI_CIE_MAX - MI_CIE_MIN) * sample;
            return { si.wavelengths, eval(si, active) * (MI_CIE_MAX - MI_CIE_MIN) };
        } else {
            DRJIT_MARK_USED(sample);
            UnpolarizedSpectrum value = eval(_si, active);
            return { dr::empty<Wavelength>(), value };
        }
    }

    Float mean() const override {
        if constexpr (is_spectral_v<Spectrum>)
            return dr::mean(srgb_model_mean(m_value));
        else
            return dr::mean(dr::mean(m_value));
    }

    ScalarFloat max() const override {
        if constexpr (is_spectral_v<Spectrum>)
            return dr::max_nested(srgb_model_mean(m_value));
        else
            return dr::max_nested(m_value);
    }

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AVXReflectanceSpectrum[" << std::endl
            << "  value = " << string::indent(m_value) << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    /**
     * Depending on the compiled variant, this plugin either stores coefficients
     * for a spectral upsampling model, or a plain RGB/monochromatic value.
     */

    Color<Float, nr_channels_v<Spectrum>> m_value;
};

MI_IMPLEMENT_CLASS_VARIANT(AVXReflectanceSpectrum, Texture)
MI_EXPORT_PLUGIN(AVXReflectanceSpectrum, "AVX spectrum")
NAMESPACE_END(mitsuba)
