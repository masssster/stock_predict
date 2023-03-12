import streamlit as st
from streamlit_option_menu import option_menu

import home,predict,contact

st.set_page_config(
    page_title = 'Skripsi Web : Stock Predict',
    layout= "wide",
)

def handle_active_page(active_page):
    if active_page == 'Home':
        home.launch()
    elif active_page == 'Predict':
        predict.launch()       
    elif active_page == 'Contact':
        contact.launch()

active_page = option_menu(
    menu_title = 'Stock Predict',

    options = ['Home', 'Predict', 'Contact'],
    icons = ['house', 'graph-up-arrow',  'telephone-fill'],
    default_index = 0,
    orientation = 'horizontal',  
    styles={
        "container": {"padding": "0!important", "background-color": "white", "margin": "0px!important", },
        "icon": {"color": "orange", "font-size": "25px"}, 
        "nav-link": {"font-size": "25px", "text-align": "left", "margin":"0px", "--hover-color": "#eee"},
        "nav-link-selected": {"background-color": "grey"},
    }
)

handle_active_page(active_page)