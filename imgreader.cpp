/*===================================================================

The Medical Imaging Interaction Toolkit (MITK)

Copyright (c) German Cancer Research Center,
Division of Medical and Biological Informatics.
All rights reserved.

This software is distributed WITHOUT ANY WARRANTY; without
even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE.

See LICENSE.txt or http://www.mitk.org for details.

===================================================================*/

#include <mitkDataNodeFactory.h>
#include <mitkStandaloneDataStorage.h>

#include <stdexcept>
#include <cstdio>

// Load image (nrrd format) and display it in a 2D view
int main(int argc, char* argv[]) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <filename>\n", argv[0]);
        return 1;
    }

    mitk::StandaloneDataStorage::Pointer ds = mitk::StandaloneDataStorage::New();

    mitk::DataNodeFactory::Pointer reader=mitk::DataNodeFactory::New();
    const char * filename = argv[1];
    try
    {
        reader->SetFileName(filename);
        reader->Update();
        //*************************************************************************
        // Part III: Put the data into the datastorage
        //*************************************************************************

        // Add the node to the DataStorage
        ds->Add(reader->GetOutput());
    }
    catch(std::exception &exc) {
        fprintf(stderr, "Could not open file %s: %s\n", filename, exc.what());
        return -1;
    }

}

