#include <mitsuba/core/bitmap.h>
#include <mitsuba/core/filesystem.h>
#include <mitsuba/core/fstream.h>
#include <mitsuba/core/spectrum.h>
#include <mitsuba/core/string.h>
#include <mitsuba/render/film.h>
#include <mitsuba/render/fwd.h>
#include <mitsuba/render/imageblock.h>

#include <mutex>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _film-hdrfilm:

High dynamic range film (:monosp:`hdrfilm`)
-------------------------------------------

.. pluginparameters::
 :extra-rows: 7

 * - width, height
   - |int|
   - Width and height of the camera sensor in pixels. Default: 768, 576)

 * - file_format
   - |string|
   - Denotes the desired output file format. The options are :monosp:`openexr`
     (for ILM's OpenEXR format), :monosp:`rgbe` (for Greg Ward's RGBE format), or
     :monosp:`pfm` (for the Portable Float Map format). (Default: :monosp:`openexr`)

 * - pixel_format
   - |string|
   - Specifies the desired pixel format of output images. The options are :monosp:`luminance`,
     :monosp:`luminance_alpha`, :monosp:`rgb`, :monosp:`rgba`, :monosp:`xyz` and :monosp:`xyza`.
     (Default: :monosp:`rgb`)

 * - component_format
   - |string|
   - Specifies the desired floating  point component format of output images (when saving to disk).
     The options are :monosp:`float16`, :monosp:`float32`, or :monosp:`uint32`.
     (Default: :monosp:`float16`)

 * - crop_offset_x, crop_offset_y, crop_width, crop_height
   - |int|
   - These parameters can optionally be provided to select a sub-rectangle
     of the output. In this case, only the requested regions
     will be rendered. (Default: Unused)

 * - sample_border
   - |bool|
   - If set to |true|, regions slightly outside of the film plane will also be sampled. This may
     improve the image quality at the edges, especially when using very large reconstruction
     filters. In general, this is not needed though. (Default: |false|, i.e. disabled)

 * - compensate
   - |bool|
   - If set to |true|, sample accumulation will be performed using Kahan-style
     error-compensated accumulation. This can be useful to avoid roundoff error
     when accumulating very many samples to compute reference solutions using
     single precision variants of Mitsuba. This feature is currently only supported
     in JIT variants and can make sample accumulation quite a bit more expensive.
     (Default: |false|, i.e. disabled)

 * - (Nested plugin)
   - :paramtype:`rfilter`
   - Reconstruction filter that should be used by the film. (Default: :monosp:`gaussian`, a windowed
     Gaussian filter)

 * - size
   - ``Vector2u``
   - Width and height of the camera sensor in pixels
   - |exposed|

 * - crop_size
   - ``Vector2u``
   - Size of the sub-rectangle of the output in pixels
   - |exposed|

 * - crop_offset
   - ``Point2u``
   - Offset of the sub-rectangle of the output in pixels
   - |exposed|

This is the default film plugin that is used when none is explicitly specified. It stores the
captured image as a high dynamic range OpenEXR file and tries to preserve the rendering as much as
possible by not performing any kind of post processing, such as gamma correction---the output file
will record linear radiance values.

When writing OpenEXR files, the film will either produce a luminance, luminance/alpha, RGB(A),
or XYZ(A) tristimulus bitmap having a :monosp:`float16`,
:monosp:`float32`, or :monosp:`uint32`-based internal representation based on the chosen parameters.
The default configuration is RGB with a :monosp:`float16` component format, which is appropriate for
most purposes.

For OpenEXR files, Mitsuba 3 also supports fully general multi-channel output; refer to
the :ref:`aov <integrator-aov>` or :ref:`stokes <integrator-stokes>` plugins for
details on how this works.

The plugin can also write RLE-compressed files in the Radiance RGBE format pioneered by Greg Ward
(set :monosp:`file_format=rgbe`), as well as the Portable Float Map format
(set :monosp:`file_format=pfm`). In the former case, the :monosp:`component_format` and
    :monosp:`pixel_format` parameters are ignored, and the output is :monosp:`float8`-compressed RGB
                                              data. PFM output is restricted to :monosp:`float32`-valued images using the :monosp:`rgb` or
    :monosp:`luminance` pixel formats. Due to the superior accuracy and adoption of OpenEXR, the use of
        these two alternative formats is discouraged however.

    When RGB(A) output is selected, the measured spectral power distributions are
        converted to linear RGB based on the CIE 1931 XYZ color matching curves and
        the ITU-R Rec. BT.709-3 primaries with a D65 white point.

The following XML snippet describes a film that writes a full-HD RGBA OpenEXR file:

 .. tabs::
    .. code-block:: xml

        <film type="avxfilm">
            <string name="pixel_format" value="rgba"/>
            <integer name="width" value="1920"/>
            <integer name="height" value="1080"/>
        </film>

    .. code-tab:: python

        'type': 'hdrfilm',
        'pixel_format': 'rgba',
        'width': 1920,
        'height': 1080

 */

template <typename Float, typename Spectrum>
class AVXFilm final : public Film<Float, Spectrum> {
public:
    MI_IMPORT_BASE(Film, m_size, m_crop_size, m_crop_offset, m_sample_border,
                   m_filter, m_flags)
    MI_IMPORT_TYPES(ImageBlock)

    AVXFilm(const Properties &props) : Base(props) {
        std::string component_format = string::to_lower(
            props.string("component_format", "float16"));

        m_file_format = Bitmap::FileFormat::OpenEXR;
        m_pixel_format = Bitmap::PixelFormat::MultiChannel;
        m_flags = +FilmFlags::Alpha;

        if (component_format == "float16")
            m_component_format = Struct::Type::Float16;
        else if (component_format == "float32")
            m_component_format = Struct::Type::Float32;
        else if (component_format == "uint32")
            m_component_format = Struct::Type::UInt32;
        else
            Throw("The \"component_format\" parameter must either be "
                  "equal to \"float16\", \"float32\", or \"uint32\"."
                  " Found %s instead.", component_format);

        m_compensate = props.get<bool>("compensate", false);

        props.mark_queried("banner"); // no banner in Mitsuba 3
    }

    size_t prepare(const std::vector<std::string> &aovs) override {
        bool alpha = has_flag(m_flags, FilmFlags::Alpha);
        size_t base_channels = alpha ? 10 : 9;

        std::vector<std::string> channels(base_channels + aovs.size());

        for (size_t i = 0; i < 8; ++i)
            channels[i] = "i" + std::to_string(i);

        if (alpha) {
            channels[8] = "A";
            channels[9] = "W";
        } else {
            channels[8] = "W";
        }

        for (size_t i = 0; i < aovs.size(); ++i)
            channels[base_channels + i] = aovs[i];

        /* locked */ {
            std::lock_guard<std::mutex> lock(m_mutex);
            m_storage = new ImageBlock(m_crop_size, m_crop_offset,
                                        (uint32_t) channels.size());
            m_channels = channels;
        }

        std::sort(channels.begin(), channels.end());
        auto it = std::unique(channels.begin(), channels.end());
        if (it != channels.end())
            Throw("Film::prepare(): duplicate channel name \"%s\"", *it);

        return m_channels.size();
    }

    ref<ImageBlock> create_block(const ScalarVector2u &size, bool normalize,
                                 bool border) override {
        bool warn = !dr::is_jit_v<Float> && !is_spectral_v<Spectrum>;

        bool default_config = size == ScalarVector2u(0);

        return new ImageBlock(default_config ? m_crop_size : size,
                              default_config ? m_crop_offset : ScalarPoint2u(0),
                              (uint32_t) m_channels.size(), m_filter.get(),
                              border /* border */,
                              normalize /* normalize */,
                              dr::is_jit_v<Float> /* coalesce */,
                              m_compensate /* compensate */,
                              warn /* warn_negative */,
                              warn /* warn_invalid */);
    }

    void put_block(const ImageBlock *block) override {
        Assert(m_storage != nullptr);
        std::lock_guard<std::mutex> lock(m_mutex);
        m_storage->put_block(block);
    }

    TensorXf develop(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        if (raw) {
            std::lock_guard<std::mutex> lock(m_mutex);
            return m_storage->tensor();
        }

        if constexpr (dr::is_jit_v<Float>) {
            Float data;
            uint32_t source_ch;
            uint32_t pixel_count;
            ScalarVector2i size;

            /* locked */ {
                std::lock_guard<std::mutex> lock(m_mutex);
                data        = m_storage->tensor().array();
                size        = m_storage->size();
                source_ch   = (uint32_t) m_storage->channel_count();
                pixel_count = dr::prod(m_storage->size());
            }

            /* The following code develops weighted image block data into
               an output image of the desired configuration, while using
               a minimal number of JIT kernel launches. */

            // Number of arbitrary output variables (AOVs)
            bool alpha = has_flag(m_flags, FilmFlags::Alpha);
            uint32_t base_ch = alpha ? 10 : 9,
                     aovs    = source_ch - base_ch;

            // Number of channels of the target tensor
            uint32_t target_ch = aovs + (uint32_t) alpha;

            // Index vectors referencing pixels & channels of the output image
            UInt32 idx         = dr::arange<UInt32>(pixel_count * target_ch),
                   pixel_idx   = idx / target_ch,
                   channel_idx = dr::fmadd(pixel_idx, uint32_t(-(int) target_ch), idx);

            /* Index vectors referencing source pixels/weights as follows:
                 values_idx = R1, G1, B1, R2, G2, B2 (for RGB output)
                 weight_idx = W1, W1, W1, W2, W2, W2 */
            UInt32 values_idx = dr::fmadd(pixel_idx, source_ch, channel_idx),
                   weight_idx = dr::fmadd(pixel_idx, source_ch, base_ch - 1);

            // If AOVs are desired, their indices in 'values_idx' must be shifted
            if (aovs) {
                // Index of first AOV channel in output image
                uint32_t first_aov = 8 + (uint32_t) alpha;
                values_idx[channel_idx >= first_aov] += base_ch - first_aov;
            }

            Mask value_mask = true;

            // Gather the pixel values from the image data buffer
            Float weight = dr::gather<Float>(data, weight_idx),
                  values = dr::gather<Float>(data, values_idx, value_mask);

            // Perform the weight division unless the weight is zero
            values /= dr::select(dr::eq(weight, 0.f), 1.f, weight);

            size_t shape[3] = { (size_t) size.y(), (size_t) size.x(),
                                target_ch };

            return TensorXf(values, 3, shape);
        } else {
            ref<Bitmap> source = bitmap();
            ScalarVector2i size = source->size();
            size_t width = source->channel_count() * dr::prod(size);
            auto data = dr::load<DynamicBuffer<ScalarFloat>>(source->data(), width);

            size_t shape[3] = { (size_t) source->height(),
                                (size_t) source->width(),
                                source->channel_count() };

            return TensorXf(data, 3, shape);
        }
    }


    ref<Bitmap> bitmap(bool raw = false) const override {
        if (!m_storage)
            Throw("No storage allocated, was prepare() called first?");

        std::lock_guard<std::mutex> lock(m_mutex);
        auto &&storage = dr::migrate(m_storage->tensor().array(), AllocType::Host);

        if constexpr (dr::is_jit_v<Float>)
            dr::sync_thread();

        ref<Bitmap> source = new Bitmap(Bitmap::PixelFormat::MultiChannel, struct_type_v<ScalarFloat>,
                                        m_storage->size(), m_storage->channel_count(),
                                        m_channels, (uint8_t *) storage.data());

        if (!raw)
            Log(Warn, "Only raw mode is available for AVX film!");
        return source;
    }

    void write(const fs::path &path) const override {
        fs::path filename = path;
        std::string proper_extension;
        proper_extension = ".exr";

        std::string extension = string::to_lower(filename.extension().string());
        if (extension != proper_extension)
            filename.replace_extension(proper_extension);

        #if !defined(_WIN32)
            Log(Info, "\U00002714  Developing \"%s\" ..", filename.string());
        #else
            Log(Info, "Developing \"%s\" ..", filename.string());
        #endif

        ref<Bitmap> source = bitmap();
        if (m_component_format != struct_type_v<ScalarFloat>) {
            // Mismatch between the current format and the one expected by the film
            // Conversion is necessary before saving to disk
            std::vector<std::string> channel_names;
            for (size_t i = 0; i < source->channel_count(); i++)
                channel_names.push_back(source->struct_()->operator[](i).name);
            ref<Bitmap> target = new Bitmap(
                source->pixel_format(),
                m_component_format,
                source->size(),
                source->channel_count(),
                channel_names);
            source->convert(target);

            target->write(filename, m_file_format);
        } else {
            source->write(filename, m_file_format);
        }
    }

    void schedule_storage() override {
        dr::schedule(m_storage->tensor());
    };

    std::string to_string() const override {
        std::ostringstream oss;
        oss << "AVXFilm[" << std::endl
            << "  size = " << m_size << "," << std::endl
            << "  crop_size = " << m_crop_size << "," << std::endl
            << "  crop_offset = " << m_crop_offset << "," << std::endl
            << "  sample_border = " << m_sample_border << "," << std::endl
            << "  compensate = " << m_compensate << "," << std::endl
            << "  filter = " << m_filter << "," << std::endl
            << "  file_format = " << m_file_format << "," << std::endl
            << "  pixel_format = " << m_pixel_format << "," << std::endl
            << "  component_format = " << m_component_format << "," << std::endl
            << "]";
        return oss.str();
    }

    MI_DECLARE_CLASS()
protected:
    Bitmap::FileFormat m_file_format;
    Bitmap::PixelFormat m_pixel_format;
    Struct::Type m_component_format;
    bool m_compensate;
    ref<ImageBlock> m_storage;
    mutable std::mutex m_mutex;
    std::vector<std::string> m_channels;
};

MI_IMPLEMENT_CLASS_VARIANT(AVXFilm, Film)
MI_EXPORT_PLUGIN(AVXFilm, "AVX Film")
NAMESPACE_END(mitsuba)
