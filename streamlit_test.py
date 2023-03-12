###Imports
import streamlit as st
import base64
from PIL import Image
import numpy as np
import pydeck as pdk
import pandas as pd
#import urllib.error


### Paths
PATH_BACKGROUND_IMAGE = r"webApp/hydro.jpg"

@st.cache(allow_output_mutation=True)
def get_base64_of_bin_file(bin_file):
    with open(bin_file,'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def from_data_file(filename):
    url = (
        "http://raw.githubusercontent.com/streamlit/"
        "example-data/master/hello/v1/%s" % filename
    )
    return pd.read_json(url)


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
    return


def add_logo_2(png_file):
    '''
    processes the logo of the FoodiX startup
    '''
    width = 350
    height = 150
    logo = Image.open(png_file)
    modified_logo = logo.resize((width,height))
    return modified_logo


def add_carbon(png_file):
    '''
    processes the logo of the FoodiX startup
    '''
    width = 200
    height = 200
    logo = Image.open(png_file)
    modified_logo = logo.resize((width,height))
    return modified_logo


def page_intro():
    '''
    main page with background and sidebar
    '''
    ###Imports
    import streamlit as st
    import base64
    from PIL import Image

    st.markdown('<h1 style="color:black;">Hydrogene truck network optimization Project</h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:black;">On this page you can test our digital solutions </h3>', unsafe_allow_html=True)
    #background image
    set_png_as_page_bg(PATH_BACKGROUND_IMAGE)

    # Add a sidebar to the web page.
    st.markdown('---')
    #st.sidebar.image(add_logo_2(PATH_FOODIX_LOGO))
    st.sidebar.markdown('Project led with AirLiquide and the french Ministry of Transports')
    st.sidebar.markdown('We help you develop an hydrogen truck network across France')
    st.sidebar.markdown('---')
    st.sidebar.write('Developed by Group 7')
    st.sidebar.write('- Cesar Bareau')
    st.sidebar.write('- Augustin De La Brosse')
    st.sidebar.write('- Alexandra Giraud')
    st.sidebar.write('- Camille Keisser')
    st.sidebar.write('- Charlotte Simon')
    st.sidebar.markdown('---')


def page_1():
    '''
    page 1
    '''
    ###Imports
    import streamlit as st
    import base64

    #display
    st.markdown('---')
    st.markdown('<h1 style="color:white;">Page 1</h1>', unsafe_allow_html=True)
    st.markdown('<h2 style="color:white;">Page à définir</h2>', unsafe_allow_html=True)
    set_png_as_page_bg(PATH_BACKGROUND_IMAGE)

    #image upload & processing
    upload = st.file_uploader('Insert file')


# def page_2():
#     '''
#     page 2
#     '''
#     import pydeck as pdk

#     ALL_LAYERS = {
#         "Bike Rentals": pdk.Layer(
#             "HexagonLayer",
#             data=from_data_file("bike_rental_stats.json"),
#             get_position=["lon", "lat"],
#             radius=200,
#             elevation_scale=4,
#             elevation_range=[0, 1000],
#             extruded=True,
#         ),
#         "Bart Stop Exits": pdk.Layer(
#             "ScatterplotLayer",
#             data=from_data_file("bart_stop_stats.json"),
#             get_position=["lon", "lat"],
#             get_color=[200, 30, 0, 160],
#             get_radius="[exits]",
#             radius_scale=0.05,
#         ),
#         "Bart Stop Names": pdk.Layer(
#             "TextLayer",
#             data=from_data_file("bart_stop_stats.json"),
#             get_position=["lon", "lat"],
#             get_text="name",
#             get_color=[0, 0, 0, 200],
#             get_size=15,
#             get_alignment_baseline="'bottom'",
#         ),
#         "Outbound Flow": pdk.Layer(
#             "ArcLayer",
#             data=from_data_file("bart_path_stats.json"),
#             get_source_position=["lon", "lat"],
#             get_target_position=["lon2", "lat2"],
#             get_source_color=[200, 30, 0, 160],
#             get_target_color=[200, 30, 0, 160],
#             auto_highlight=True,
#             width_scale=0.0001,
#             get_width="outbound",
#             width_min_pixels=3,
#             width_max_pixels=30,
#         ),
#     }
#     st.sidebar.markdown("### Map Layers")
#     selected_layers = [
#         layer
#         for layer_name, layer in ALL_LAYERS.items()
#         if st.sidebar.checkbox(layer_name, True)
#     ]
#     if selected_layers:
#         st.pydeck_chart(
#             pdk.Deck(
#                 map_style=None,
#                 initial_view_state={
#                     "latitude": 37.76,
#                     "longitude": -122.4,
#                     "zoom": 11,
#                     "pitch": 50,
#                 },
#                 layers=selected_layers,
#             )
#         )
#     else:
#         st.error("Please choose at least one layer above.")
#     # except URLError as e:
#     #     st.error(
#     #         """
#     #         **This demo requires internet access.**
#     #         Connection error: %s
#     #     """
#     #         % e.reason
#     #     )

def page_3():
    '''
    page 3 with coordinate
    '''
    ###Imports
    import streamlit as st
    import base64
    from PIL import Image
    import cv2
    import numpy as np
    import leafmap.foliumap as leafmap

    #display

    st.markdown('---')
    st.markdown('<h1 style="color:white;"> Network </h1>', unsafe_allow_html=True)
    st.markdown('<h3 style="color:white;"> please choose an area on the map</h3>', unsafe_allow_html=True)
    set_png_as_page_bg(PATH_BACKGROUND_IMAGE)


    ##leafmap
    m = leafmap.Map(center = [0,0],zoom = 2)
    #in_geojson = 'https://raw.githubusercontent.com/giswqs/leafmap/master/examples/data/cable-geo.geojson'
    #m.add_geojson(in_geojson, layer_name="Cable lines")
    m.to_streamlit()


page_names_to_funcs = {
    "Welcome Page": page_intro,
    " Demo": page_1,
    "map test" : page_3
}

demo_name = st.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()


#if __name__ == __main__():

