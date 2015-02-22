/*
 * $File: mitkconv.cpp
 * $Date: Sun Feb 22 21:09:44 2015 +0800
 * $Author: jiakai <jia.kai66@gmail.com>
 */

#include "mitkSceneIO.h"
#include "mitkNodePredicateDataType.h"
#include "mitkImage.h"
#include "mitkSurface.h"
#include "mitkImageReadAccessor.h"

#include <stdexcept>
#include <typeinfo>
#include <cstdlib>
#include <cstdio>

#define AUTO_DEF(name, v) \
    __typeof__(v) name = v


// see mitkPythonService.cpp

static void work_on_image(FILE *fout, FILE *fout_data, mitk::Image *img) {
    unsigned int *dim = img->GetDimensions();
    const mitk::Vector3D spacing = img->GetGeometry()->GetSpacing();
    const mitk::Point3D origin = img->GetGeometry()->GetOrigin();
    mitk::PixelType pixelType = img->GetPixelType();
    itk::ImageIOBase::IOPixelType ioPixelType =
        img->GetPixelType().GetPixelType();
    mitk::ImageReadAccessor racc(img);
    void* array = (void*) racc.GetData();

    if (ioPixelType != itk::ImageIOBase::SCALAR)
        throw std::runtime_error("bad ioPixelType");

    printf("type=%d\n", pixelType.GetComponentType());
    const char *pixel_t_text;
    size_t pixel_t_size;
    switch (pixelType.GetComponentType()) {
        case itk::ImageIOBase::UCHAR:
            pixel_t_text = "uint8";
            pixel_t_size = 1;
            break;
        case itk::ImageIOBase::SHORT:
            pixel_t_text = "int16";
            pixel_t_size = 2;
            break;
        default:
            throw std::runtime_error("unkown pixel type");
    }

    fprintf(fout, "dim: %u %u %u\n", dim[0], dim[1], dim[2]);
    fprintf(fout, "spacing: %.5e %.5e %.5e\n",
            spacing[0], spacing[1], spacing[2]);
    fprintf(fout, "origin: %.5e %.5e %.5e\n", origin[0], origin[1], origin[2]);
    fprintf(fout, "offset: %zd\n", ftell(fout_data));
    fprintf(fout, "pixel_type: %s\n", pixel_t_text);
    fwrite(array, pixel_t_size, dim[0] * dim[1] * dim[2], fout_data);
}

int main(int argc, char* argv[]) {
    if (argc < 4) {
        fprintf(stderr, "usage: "
                "%s <meta output file name> <data output file name>"
                " <input files...>\n", argv[0]);
        return 1;
    }

    FILE *fout_meta = fopen(argv[1], "w"),
         *fout_data = fopen(argv[2], "wb");

    static char realpath_buf[PATH_MAX];
    fprintf(fout_meta, "fpath_data: %s\n", realpath(argv[2], realpath_buf));

    for (int i = 3; i < argc; i ++) {
        const char *fname = argv[i];
        mitk::SceneIO::Pointer sceneIO = mitk::SceneIO::New();
        mitk::DataStorage::Pointer ds = sceneIO->LoadScene(fname);
        mitk::DataStorage::SetOfObjects::ConstPointer rs = ds->GetSubset(NULL);
        printf("%s: nr_node=%zd\n", fname, rs->size());

        AUTO_DEF(pred_img, mitk::TNodePredicateDataType<mitk::Image>::New());
        AUTO_DEF(pred_surface,
                mitk::TNodePredicateDataType<mitk::Surface>::New());

        fprintf(fout_meta, "file: %s\n", realpath(fname, realpath_buf));

        for (__typeof__(rs->begin()) iter = rs->begin();
                iter != rs->end(); iter ++) {
            mitk::DataNode *node = *iter;
            AUTO_DEF(i, *iter);
            std::string name;
            i->GetName(name);

            bool is_img = pred_img->CheckNode(node),
                 is_surface = pred_surface->CheckNode(node);
            printf("name=%s is_image=%d is_surface=%d\n",
                    name.c_str(), is_img, is_surface);
            if (!(is_img ^ is_surface)) {
                printf("WTF!\n");
                return -1;
            }
            if (is_img) {
                fprintf(fout_meta, "image: %s\n", name.c_str());
                work_on_image(fout_meta, fout_data,
                        dynamic_cast<mitk::Image*>(node->GetData()));
                fprintf(fout_meta, "end: %s\n", name.c_str());
            }
        }
    }

    fclose(fout_meta);
    fclose(fout_data);
}

// vim: syntax=cpp.doxygen foldmethod=marker foldmarker=f{{{,f}}}
