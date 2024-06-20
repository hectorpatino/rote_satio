import warnings

import numpy as np

from spyndex import spyndex
from spyndex import constants
import xarray as xr

from rote_satio.utils.base_transformer import BaseIOTransformer


class IndexTransformer(BaseIOTransformer):
    def __init__(
            self,
            program: str = 'Planet',
    ):
        """
        This is a `spyndex` wrapper to automate the computation of multiple spectral indexes.
        It is heavily based on the `spyndex` package, which is a Python package for the computation of spectral indices
        for more information please visit [spyndex](https://spyndex.readthedocs.io/en/latest/):

        Args:
            program: Program of the sensor of the image. Until 0.0.2beta version only 'Planet' is supported.

        """
        self.program = program
        self.indexes = self.get_indexes()
        self._check_program(self.program)



    def get_indexes(self):
        """
        It returns the indexes supported by the program of the program being used by `spyndex
        Returns:
            List of indexes supported by the program.

        """
        # skiped indexes due to errors.
        skip_indexes = ['NIRvP', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI', 'kVARI', 'AVI']
        if self.program == 'Planet':
            self.indexes = [
                'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI', 'bNIRv', 'CIG', 'CVI', 'DSWI4', 'DVI',
                'EBI', 'ENDVI', 'EVI', 'EVI2', 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI',
                'GBNDVI', 'GCC', 'GDVI', 'GEMI', 'GLI', 'GNDVI', 'GOSAVI', 'GRNDVI', 'GRVI', 'GSAVI', 'IAVI', 'IKAW',
                'IPVI', 'MCARI1', 'MCARI2', 'MGRVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSR', 'MTVI1', 'MTVI2',
                'NDDI', 'NDVI', 'NDYI', 'NGRDI', 'NIRv', 'NIRvP', 'NLI', 'NormG', 'NormNIR',
                'NormR', 'OCVI', 'OSAVI', 'RCC', 'RDVI', 'RGBVI', 'RGRI', 'RI', 'SARVI', 'SAVI', 'SAVI2', 'SEVI',
                'SI', 'SR', 'SR2', 'TDVI', 'TGI', 'TSAVI', 'TVI', 'TriVI', 'VARI', 'VIG', 'WDRVI', 'WDVI', 'NDTI',
                'NDWI', 'NDWIns', 'OSI', 'PI', 'RNDVI', 'BAI', 'NDGlaI', 'NDSII', 'PISI', 'VgNIRBI',
                'VrNIRBI', 'BITM', 'BIXS', 'RI4XS', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI', 'kVARI']

        if self.program == 'Landsat-TM':
            warnings.warn("Landsat-TM is not fully supported yet as some indexes may be broken."
                          "if thats the case please add to `skip_indexes` the ones that are causing troubles and"
                          "create a issue.", UserWarning)
            self.indexes = ['AFRI1600', 'AFRI2100', 'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI', 'bNIRv', 'CIG',
                            'CVI', 'DSI', 'DSWI1', 'DSWI2', 'DSWI3', 'DSWI4', 'DSWI5', 'DVI', 'DVIplus', 'EBI', 'ENDVI',
                            'EVI', 'EVI2', 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI', 'GBNDVI', 'GCC', 'GDVI',
                            'GEMI', 'GLI', 'GNDVI', 'GOSAVI', 'GRNDVI', 'GRVI', 'GSAVI', 'GVMI', 'IAVI', 'IKAW', 'IPVI',
                            'MCARI1', 'MCARI2', 'MGRVI', 'MNDVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSI', 'MSR', 'MTVI1',
                            'MTVI2', 'NDDI', 'NDGI', 'NDII', 'NDMI', 'NDPI', 'NDVI', 'NDYI', 'NGRDI', 'NIRv', 'NIRvH2',
                            'NIRvP', 'NLI', 'NMDI', 'NRFIg', 'NRFIr', 'NormG', 'NormNIR', 'NormR', 'OCVI', 'OSAVI',
                            'RCC', 'RDVI', 'RGBVI', 'RGRI', 'RI', 'SARVI', 'SAVI', 'SAVI2', 'SEVI', 'SI', 'SLAVI', 'SR',
                            'SR2', 'sNIRvLSWI', 'sNIRvNDPI', 'sNIRvNDVILSWIP', 'sNIRvNDVILSWIS', 'sNIRvSWIR', 'TDVI',
                            'TGI', 'TSAVI', 'TVI', 'TriVI', 'VARI', 'VIG', 'WDRVI', 'WDVI', 'ANDWI', 'AWEInsh',
                            'AWEIsh', 'FAI', 'LSWI', 'MBWI', 'MLSWI26', 'MLSWI27', 'MNDWI', 'MuWIR', 'NDPonI', 'NDTI',
                            'NDVIMNDWI', 'NDWI', 'NDWIns', 'NWI', 'OSI', 'PI', 'RNDVI', 'SWM', 'WI1', 'WI2', 'WI2015',
                            'WRI', 'BAI', 'BAIM', 'CSI', 'CSIT', 'MIRBI', 'NBR', 'NBR2', 'NBRSWIR', 'NBRT1', 'NBRT2',
                            'NBRT3', 'NDSWIR', 'NDVIT', 'NSTv1', 'NSTv2', 'SAVIT', 'VI6T', 'NBSIMS', 'NDGlaI', 'NDSI',
                            'NDSII', 'NDSInw', 'NDSaII', 'S3', 'SWI', 'BLFEI', 'BRBA', 'EBBI', 'IBI', 'NBAI', 'NBUI',
                            'NDBI', 'NDISIb', 'NDISIg', 'NDISImndwi', 'NDISIndwi', 'NDISIr', 'PISI', 'UI', 'VIBI',
                            'VgNIRBI', 'VrNIRBI', 'BI', 'BITM', 'BIXS', 'BaI', 'DBSI', 'EMBI', 'MBI', 'NBLI', 'NDBaI',
                            'NDSoI', 'NSDS', 'NSDSI1', 'NSDSI2', 'NSDSI3', 'RI4XS', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI',
                            'kVARI']
        if self.program == 'MODIS':
            warnings.warn("MODIS is not fully supported yet as some indexes may be broken."
                          "if thats the case please add to `skip_indexes` the ones that are causing troubles and"
                          "create a issue.", UserWarning)
            self.indexes = ['AFRI1600', 'AFRI2100', 'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI', 'bNIRv', 'CCI',
                            'CIG', 'CVI', 'DSI', 'DSWI1', 'DSWI2', 'DSWI3', 'DSWI4', 'DSWI5', 'DVI', 'DVIplus', 'EBI',
                            'ENDVI', 'EVI', 'EVI2', 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI', 'GBNDVI', 'GCC',
                            'GDVI', 'GEMI', 'GLI', 'GNDVI', 'GOSAVI', 'GRNDVI', 'GRVI', 'GSAVI', 'GVMI', 'IAVI', 'IKAW',
                            'IPVI', 'MCARI1', 'MCARI2', 'MGRVI', 'MNDVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSI', 'MSR',
                            'MTVI1', 'MTVI2', 'NDDI', 'NDGI', 'NDII', 'NDMI', 'NDPI', 'NDVI', 'NDYI', 'NGRDI', 'NIRv',
                            'NIRvH2', 'NIRvP', 'NLI', 'NMDI', 'NRFIg', 'NRFIr', 'NormG', 'NormNIR', 'NormR', 'OCVI',
                            'OSAVI', 'RCC', 'RDVI', 'RGBVI', 'RGRI', 'RI', 'SARVI', 'SAVI', 'SAVI2', 'SEVI', 'SI',
                            'SLAVI', 'SR', 'SR2', 'sNIRvLSWI', 'sNIRvNDPI', 'sNIRvNDVILSWIP', 'sNIRvNDVILSWIS',
                            'sNIRvSWIR', 'TDVI', 'TGI', 'TSAVI', 'TVI', 'TriVI', 'VARI', 'VIG', 'WDRVI', 'WDVI',
                            'ANDWI', 'AWEInsh', 'AWEIsh', 'FAI', 'LSWI', 'MBWI', 'MLSWI26', 'MLSWI27', 'MNDWI', 'MuWIR',
                            'NDPonI', 'NDTI', 'NDVIMNDWI', 'NDWI', 'NDWIns', 'NWI', 'OSI', 'PI', 'RNDVI', 'SWM', 'WI1',
                            'WI2', 'WI2015', 'WRI', 'BAI', 'BAIM', 'CSI', 'MIRBI', 'NBR', 'NBR2', 'NBRSWIR', 'NDSWIR',
                            'NBSIMS', 'NDGlaI', 'NDSI', 'NDSII', 'NDSInw', 'NDSaII', 'S3', 'SWI', 'BLFEI', 'BRBA',
                            'IBI', 'NBAI', 'NDBI', 'PISI', 'UI', 'VIBI', 'VgNIRBI', 'VrNIRBI', 'BI', 'BITM', 'BIXS',
                            'BaI', 'DBSI', 'EMBI', 'MBI', 'NDSoI', 'NSDS', 'NSDSI1', 'NSDSI2', 'NSDSI3', 'RI4XS',
                            'kEVI', 'kIPVI', 'kNDVI', 'kRVI', 'kVARI']
        if self.program == 'Sentinel-2':
            warnings.warn("Sentinel-2 is not fully supported yet as some indexes may be broken."
                          "if thats the case please add to `skip_indexes` the ones that are causing troubles and"
                          "create a issue.", UserWarning)
            self.indexes = ['AFRI1600', 'AFRI2100', 'ARI', 'ARI2', 'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI',
                            'bNIRv', 'CIG', 'CIRE', 'CVI', 'DSI', 'DSWI1', 'DSWI2', 'DSWI3', 'DSWI4', 'DSWI5', 'DVI',
                            'DVIplus', 'EBI', 'ENDVI', 'EVI', 'EVI2', 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI',
                            'GBNDVI', 'GCC', 'GDVI', 'GEMI', 'GLI', 'GM1', 'GM2', 'GNDVI', 'GOSAVI', 'GRNDVI', 'GRVI',
                            'GSAVI', 'GVMI', 'IAVI', 'IKAW', 'IPVI', 'IRECI', 'MCARI', 'MCARI1', 'MCARI2', 'MCARI705',
                            'MCARIOSAVI', 'MCARIOSAVI705', 'MGRVI', 'MNDVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSI', 'MSR',
                            'MSR705', 'MTCI', 'MTVI1', 'MTVI2', 'mND705', 'mSR705', 'ND705', 'NDDI', 'NDGI', 'NDII',
                            'NDMI', 'NDPI', 'NDREI', 'NDVI', 'NDVI705', 'NDYI', 'NGRDI', 'NIRv', 'NIRvH2', 'NIRvP',
                            'NLI', 'NMDI', 'NRFIg', 'NRFIr', 'NormG', 'NormNIR', 'NormR', 'OCVI', 'OSAVI', 'PSRI',
                            'RCC', 'RDVI', 'REDSI', 'RENDVI', 'RGBVI', 'RGRI', 'RI', 'RVI', 'S2REP', 'SARVI', 'SAVI',
                            'SAVI2', 'SEVI', 'SI', 'SIPI', 'SLAVI', 'SR', 'SR2', 'SR3', 'SR555', 'SR705', 'SeLI',
                            'sNIRvLSWI', 'sNIRvNDPI', 'sNIRvNDVILSWIP', 'sNIRvNDVILSWIS', 'sNIRvSWIR', 'TCARI',
                            'TCARIOSAVI', 'TCARIOSAVI705', 'TCI', 'TDVI', 'TGI', 'TRRVI', 'TSAVI', 'TTVI', 'TVI',
                            'TriVI', 'VARI', 'VARI700', 'VI700', 'VIG', 'WDRVI', 'WDVI', 'ANDWI', 'AWEInsh', 'AWEIsh',
                            'FAI', 'LSWI', 'MBWI', 'MLSWI26', 'MLSWI27', 'MNDWI', 'MuWIR', 'NDCI', 'NDPonI', 'NDTI',
                            'NDVIMNDWI', 'NDWI', 'NDWIns', 'NWI', 'OSI', 'PI', 'RNDVI', 'S2WI', 'SWM', 'TWI', 'WI1',
                            'WI2', 'WI2015', 'WRI', 'BAI', 'BAIM', 'BAIS2', 'CSI', 'MIRBI', 'NBR', 'NBR2', 'NBRSWIR',
                            'NBRplus', 'NDSWIR', 'NBSIMS', 'NDGlaI', 'NDSI', 'NDSII', 'NDSInw', 'NDSaII', 'S3', 'SWI',
                            'BLFEI', 'BRBA', 'IBI', 'NBAI', 'NDBI', 'NHFD', 'PISI', 'UI', 'VIBI', 'VgNIRBI', 'VrNIRBI',
                            'BI', 'BITM', 'BIXS', 'BaI', 'DBSI', 'EMBI', 'MBI', 'NDSoI', 'NSDS', 'NSDSI1', 'NSDSI2',
                            'NSDSI3', 'RI4XS', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI', 'kVARI']
        elif self.program == ' Landsat-ETM+':
            warnings.warn("Landsat-ETM+ is not fully supported yet as some indexes may be broken."
                          "if thats the case please add to `skip_indexes` the ones that are causing troubles and"
                          "create a issue.", UserWarning)
            self.indexes = ['AFRI1600', 'AFRI2100', 'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI', 'bNIRv', 'CIG',
                            'CVI', 'DSI', 'DSWI1', 'DSWI2', 'DSWI3', 'DSWI4', 'DSWI5', 'DVI', 'DVIplus', 'EBI', 'ENDVI',
                            'EVI', 'EVI2', 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI', 'GBNDVI', 'GCC', 'GDVI',
                            'GEMI', 'GLI', 'GNDVI', 'GOSAVI', 'GRNDVI', 'GRVI', 'GSAVI', 'GVMI', 'IAVI', 'IKAW', 'IPVI',
                            'MCARI1', 'MCARI2', 'MGRVI', 'MNDVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSI', 'MSR', 'MTVI1',
                            'MTVI2', 'NDDI', 'NDGI', 'NDII', 'NDMI', 'NDPI', 'NDVI', 'NDYI', 'NGRDI', 'NIRv', 'NIRvH2',
                            'NIRvP', 'NLI', 'NMDI', 'NRFIg', 'NRFIr', 'NormG', 'NormNIR', 'NormR', 'OCVI', 'OSAVI',
                            'RCC', 'RDVI', 'RGBVI', 'RGRI', 'RI', 'SARVI', 'SAVI', 'SAVI2', 'SEVI', 'SI', 'SLAVI', 'SR',
                            'SR2', 'sNIRvLSWI', 'sNIRvNDPI', 'sNIRvNDVILSWIP', 'sNIRvNDVILSWIS', 'sNIRvSWIR', 'TDVI',
                            'TGI', 'TSAVI', 'TVI', 'TriVI', 'VARI', 'VIG', 'WDRVI', 'WDVI', 'ANDWI', 'AWEInsh',
                            'AWEIsh', 'FAI', 'LSWI', 'MBWI', 'MLSWI26', 'MLSWI27', 'MNDWI', 'MuWIR', 'NDPonI', 'NDTI',
                            'NDVIMNDWI', 'NDWI', 'NDWIns', 'NWI', 'OSI', 'PI', 'RNDVI', 'SWM', 'WI1', 'WI2', 'WI2015',
                            'WRI', 'BAI', 'BAIM', 'CSI', 'CSIT', 'MIRBI', 'NBR', 'NBR2', 'NBRSWIR', 'NBRT1', 'NBRT2',
                            'NBRT3', 'NDSWIR', 'NDVIT', 'NSTv1', 'NSTv2', 'SAVIT', 'VI6T', 'NBSIMS', 'NDGlaI', 'NDSI',
                            'NDSII', 'NDSInw', 'NDSaII', 'S3', 'SWI', 'BLFEI', 'BRBA', 'EBBI', 'IBI', 'NBAI', 'NBUI',
                            'NDBI', 'NDISIb', 'NDISIg', 'NDISImndwi', 'NDISIndwi', 'NDISIr', 'PISI', 'UI', 'VIBI',
                            'VgNIRBI', 'VrNIRBI', 'BI', 'BITM', 'BIXS', 'BaI', 'DBSI', 'EMBI', 'MBI', 'NBLI', 'NDBaI',
                            'NDSoI', 'NSDS', 'NSDSI1', 'NSDSI2', 'NSDSI3', 'RI4XS', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI',
                            'kVARI']
        elif self.program == 'Landsat-OLI':
            warnings.warn("Landsat-OLI is not fully supported yet as some indexes may be broken."
                          "if thats the case please add to `skip_indexes` the ones that are causing troubles and"
                          "create a issue.", UserWarning)
            self.indexes = [
                ['AFRI1600', 'AFRI2100', 'ARVI', 'ATSAVI', 'AVI', 'BCC', 'BNDVI', 'BWDRVI', 'bNIRv', 'CIG', 'CVI',
                 'DSI', 'DSWI1', 'DSWI2', 'DSWI3', 'DSWI4', 'DSWI5', 'DVI', 'DVIplus', 'EBI', 'ENDVI', 'EVI', 'EVI2',
                 'EVIv', 'ExG', 'ExGR', 'ExR', 'FCVI', 'GARI', 'GBNDVI', 'GCC', 'GDVI', 'GEMI', 'GLI', 'GNDVI',
                 'GOSAVI', 'GRNDVI', 'GRVI', 'GSAVI', 'GVMI', 'IAVI', 'IKAW', 'IPVI', 'MCARI1', 'MCARI2', 'MGRVI',
                 'MNDVI', 'MNLI', 'MRBVI', 'MSAVI', 'MSI', 'MSR', 'MTVI1', 'MTVI2', 'NDDI', 'NDGI', 'NDII', 'NDMI',
                 'NDPI', 'NDVI', 'NDYI', 'NGRDI', 'NIRv', 'NIRvH2', 'NIRvP', 'NLI', 'NMDI', 'NRFIg', 'NRFIr', 'NormG',
                 'NormNIR', 'NormR', 'OCVI', 'OSAVI', 'RCC', 'RDVI', 'RGBVI', 'RGRI', 'RI', 'SARVI', 'SAVI', 'SAVI2',
                 'SEVI', 'SI', 'SIPI', 'SLAVI', 'SR', 'SR2', 'sNIRvLSWI', 'sNIRvNDPI', 'sNIRvNDVILSWIP',
                 'sNIRvNDVILSWIS', 'sNIRvSWIR', 'TDVI', 'TGI', 'TSAVI', 'TVI', 'TriVI', 'VARI', 'VIG', 'WDRVI', 'WDVI',
                 'ANDWI', 'AWEInsh', 'AWEIsh', 'FAI', 'LSWI', 'MBWI', 'MLSWI26', 'MLSWI27', 'MNDWI', 'MuWIR', 'NDPonI',
                 'NDTI', 'NDVIMNDWI', 'NDWI', 'NDWIns', 'NWI', 'OSI', 'PI', 'RNDVI', 'SWM', 'WI1', 'WI2', 'WI2015',
                 'WRI', 'BAI', 'BAIM', 'CSI', 'MIRBI', 'NBR', 'NBR2', 'NBRSWIR', 'NDSWIR', 'NBSIMS', 'NDGlaI', 'NDSI',
                 'NDSII', 'NDSInw', 'NDSaII', 'S3', 'SWI', 'BLFEI', 'BRBA', 'DBI', 'IBI', 'NBAI', 'NDBI', 'PISI', 'UI',
                 'VIBI', 'VgNIRBI', 'VrNIRBI', 'BI', 'BITM', 'BIXS', 'BaI', 'DBSI', 'EMBI', 'MBI', 'NBLIOLI', 'NDSoI',
                 'NSDS', 'NSDSI1', 'NSDSI2', 'NSDSI3', 'RI4XS', 'kEVI', 'kIPVI', 'kNDVI', 'kRVI', 'kVARI']]

        return [index for index in self.indexes if index not in skip_indexes]

    def transform(self, X: xr.DataArray) -> xr.DataArray:
        """
        It computes the indexes for the input data.
        Args:
            X: Input data.
        Returns
            xr.DataArray: A new DataArray with the indexes computed.
        """
        self._check_input(X)
        self._get_params(X)

        idx = spyndex.computeIndex(
            index=self.indexes,
            params=self.params
        )
        idx = xr.where(np.isinf(idx) | np.isnan(idx), 0, idx)
        idx = idx.rename({'index': 'band'})
        idx = xr.concat([X, idx], dim='band')
        idx.attrs['long_name'] = list(X.band.values)
        idx = idx.rio.write_crs(X.rio.crs)
        return idx

    def _get_bands_names(self):
        """
        This function returns the band names for the program being used. Until 0.0.2beta version only 'Planet' is supported.
        Returns:

        """
        if self.program == 'Planet':
            return ['B1', 'B2', 'B3', 'B4']
        elif self.program == 'Landsat-TM' or self.program == 'Landsat-ETM+':
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7']
        elif self.program == 'MODIS':
            return ['B1', 'B2', 'B3', 'B4', 'B6', 'B7', 'B11']
        elif self.program == 'Sentinel-2':
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B10', 'B11', 'B12']
        elif self.program == 'Landsat-OLI':
            return ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B10', 'B11']

    def _get_params(self, X: xr.DataArray) -> None:
        """
        Some indexes require parameters to be computed. This function sets the parameters for the indexes.
        Args:
            X: An xarray DataArray with the bands of the image.

        Returns:
            None
        """
        self.params = {
            "gamma": constants.gamma.value,
            "sla": constants.sla.value,
            "slb": constants.slb.value,
            "alpha": constants.alpha.value,
            "g": constants.g.value,
            "C1": constants.C1.value,
            "C2": constants.C2.value,
            "omega": constants.omega.value,
            "L": constants.L.value,
            "nexp": constants.nexp.value,
            "cexp": constants.cexp.value,
            "fdelta": constants.fdelta.value,
            "beta": constants.beta.value,
            "epsilon": constants.epsilon.value,
            "k": constants.k.value,
        }

        if self.program == 'Planet':
            self.params['R'] = X.isel(band=0)
            self.params['G'] = X.isel(band=1)
            self.params['B'] = X.isel(band=2)
            self.params['N'] = X.isel(band=3)

        elif self.program == 'Landsat-TM' or self.program == 'Landsat-ETM+':
            self.params['B'] = X.isel(band=0)
            self.params['G'] = X.isel(band=1)
            self.params['R'] = X.isel(band=2)
            self.params['N'] = X.isel(band=3)
            self.params['S1'] = X.isel(band=4)
            self.params['S2'] = X.isel(band=5)
            self.params['T'] = X.isel(band=6)
        elif self.program == 'MODIS':
            self.params['B'] = X.isel(band=2)
            self.params['G1'] = X.isel(band=10)
            self.params['G'] = X.isel(band=3)
            self.params['N'] = X.isel(band=1)
            self.params['S1'] = X.isel(band=5)
            self.params['S2'] = X.isel(band=6)
        elif self.program == 'Sentinel-2':
            self.params['A'] = X.isel(band=0)
            self.params['B'] = X.isel(band=1)
            self.params['G'] = X.isel(band=2)
            self.params['R'] = X.isel(band=3)
            self.params['RE1'] = X.isel(band=4)
            self.params['RE2'] = X.isel(band=5)
            self.params['RE3'] = X.isel(band=6)
            self.params['N'] = X.isel(band=7)
            self.params['NIR'] = X.isel(band=8)
            self.params['WV'] = X.isel(band=9)
            self.params['S1'] = X.isel(band=10)
            self.params['S2'] = X.isel(band=11)
        elif self.program == 'Landsat-OLI':
            self.params['A'] = X.isel(band=0)
            self.params['B'] = X.isel(band=1)
            self.params['G'] = X.isel(band=2)
            self.params['R'] = X.isel(band=3)
            self.params['N'] = X.isel(band=4)
            self.params['S1'] = X.isel(band=5)
            self.params['S2'] = X.isel(band=6)
            self.params['T1'] = X.isel(band=7)
            self.params['T2'] = X.isel(band=8)
