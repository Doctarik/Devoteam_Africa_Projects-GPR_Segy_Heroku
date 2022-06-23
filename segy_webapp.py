import streamlit as st
import segyio
import segysak
import pandas as pd
import xarray as xr
import itertools
import numpy as np
import matplotlib.pyplot as plt
import hvplot.xarray
import panel as pn
#from matplotlib import figure

from holoviews import opts
from bokeh.models import HoverTool
from segysak.segy import segy_loader, segy_header_scan, segy_header_scrape
from scipy.interpolate import griddata
from IPython.display import display
from IPython.display import Image
from IPython.core.display import HTML
from IPython.display import display
from matplotlib.patches import Polygon

import pathlib

st.title("Study of the subsurface of archaeological sites and the environment using the geophysical method of remote sensing based on GPR")
#st.markdown("ceci est un test")
#st.help(st.file_uploader)

uploaded_file = st.sidebar.file_uploader("Choose a Segy file")
if uploaded_file is not None:
    st.write("The filename:\t", uploaded_file.name)
    #path_file_in   = "/home/doctarik/Devoteam_Africa_Projects/GPR/segy2xyz/segydata_in/"
    #path_file_out  = "/home/doctarik/Devoteam_Africa_Projects/GPR/segy2xyz/segydata_out/"
    #filename       = path_file_in + uploaded_file.name
    filename       = "https://github.com/Doctarik/Devoteam_Africa_Projects-GPR_Segy_Heroku/blob/master/segydata_in/3D_DAT_0192_ibm_format.SGY"
    f = segyio.open(filename, ignore_geometry = True)
    # Memory map file for faster reading (especially if file is big...)
    f.mmap()
    traces      = segyio.collect(f.trace)[:]
    ntraces     = len(f.trace)
    inlines     = []
    crosslines  = []
    twt         = f.samples
    for h in f.header:
        inlines.append(h[segyio.su.iline])
        crosslines.append(h[segyio.su.xline])
    il          = inlines
    xl          = crosslines
    il_unique = np.unique(il)
    xl_unique = np.unique(xl)
    il_min, il_max = min(il_unique), max(il_unique)
    xl_min, xl_max = min(xl_unique), max(xl_unique)
    dil = min(np.unique(np.diff(il_unique)))
    dxl = min(np.unique(np.diff(xl_unique)))
    ilines = np.arange(il_min, il_max + dil, dil)
    xlines = np.arange(xl_min, xl_max + dxl, dxl)
    nil, nxl, nt = ilines.size, xlines.size, twt.size
    ilgrid, xlgrid = np.meshgrid(np.arange(nil),np.arange(nxl), indexing='ij')
    traces_indeces = np.full((nil, nxl), np.nan)
    iils = (il - il_min) // dil
    ixls = (xl - xl_min) // dxl
    traces_indeces[iils, ixls] = np.arange(ntraces)
    traces_available = np.logical_not(np.isnan(traces_indeces))
    # reorganize traces in regular grid
    d = np.zeros((nil, nxl, nt))
    d[ilgrid.ravel()[traces_available.ravel()],
        xlgrid.ravel()[traces_available.ravel()]] = traces
    #st.write(d.shape)
    seis_3d_gath_1 = segy_loader(filename, iline=189, xline=193, cdpx=181, cdpy=185, return_geometry= True)
    #print('Reading Segyfile using segy_loader')


    st.write("Extracting 3D coordinates (inline, xline, twt)")

    # the seis_3d_gath have xarray.Dataset format and data variables is missing, so we are going to make some data manipulation to seis_3d_gath using Xarray concept

    #
    # As with our original plot using Matplotlib the interactive plots we created above have no useful axis info.
    # In the plot_inl function we had to convert our NumPy array into an DataArray in order to allow us to plot it with hvplot. However, Xarray has a lot more useful functionality that makes it ideal for using with seismic data. Xarray simplifies working with multi-dimensional data and allows dimension, coordinate and attribute labels to be added to the data (segysak utilises it and Tony's tutorial on Tuesday provided some more information on the format).
    # Xarray has two data structures
    #
    # - DataArray: for a single data variable
    # - DataSet: a container for multiple Data Arrays that share the same coordinates
    #
    # The figure below (Hoyer & Hamman, 2017) illustrates the concept of a dataset containing climate data
    seis_3d_gath_2 = seis_3d_gath_1.drop_isel(twt=[456])

    # create data with dims and coordinates
    da = xr.DataArray(data = d, dims= ["iline", "xline", "twt"])
    seis_3d_gath_3=seis_3d_gath_2.assign(data=da)

    seis_3d_gath_3.data.T.plot(
        col="iline",
        col_wrap=3,  # each row has a maximum of 4 columns
    )
    seis_3d_gath_3.data.T.isel(iline=1).plot()
    seis_3d_gath_df = seis_3d_gath_3.to_dataframe()
    display(seis_3d_gath_df)

    seis_3d_gath_df_reindex = seis_3d_gath_df.reset_index()
    display(seis_3d_gath_df_reindex)

    st.write(seis_3d_gath_df_reindex)

    new_filename_2 = path_file_out + 'seisc-data.csv'

    # Set the default image size
    # opts.defaults(opts.Image(width=800, height=600))
    # vm = np.percentile(d, 95)
    # central = len(ilines) // 2
    # def plot_inl(inl):
    #     s     = f'All samples of the Trace: {inl}'
    #     idx   = inl - ilines[0]
    #     da    = xr.DataArray(d[idx,:,:].T)
    #     #st.write(da)
    #     p     = da.hvplot.image(clim=(-vm, vm), cmap='RdBu', title= s , clabel='Amplitude', flip_yaxis=True, xlabel= 'xlines', ylabel= 'twt')
    #     return p
    # st.plot_inl(ilines[central])
    # pn.interact(plot_inl, inl = ilines)

    st.write("Plotting (inline, xline, twt)")

    fig = plt.figure()
    ine=st.sidebar.slider("the inline coordinate",0,28)
    seis_3d_gath_3.data.T.isel(iline=ine).plot()
    st.pyplot(fig)

    st.write("Saving the Final Result ")
    st.write(new_filename_2)
    seis_3d_gath_df = seis_3d_gath_3.to_dataframe()
    seis_3d_gath_df.to_csv(new_filename_2, sep = ",", index=True)
    st.write("Finished saving")
