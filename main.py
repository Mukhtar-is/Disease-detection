import streamlit as st
import tensorflow as tf
import numpy as np


#Tensorflow Model Prediction
def model_prediction(test_image):
#    model = tf.keras.models.load_model("trained_maize_disease_model.keras")
    model = tf.keras.models.load_model("maize_disease_detection_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image,target_size=(256,256))

    image_arr = tf.keras.preprocessing.image.img_to_array(image)    # processing image ka cusub ee shaqayna
#    input_arr = tf.keras.preprocessing.image.img_to_array(image)      

    input_arr = np.expand_dims(image_arr, axis=0) # new convert single image to batch (waxa u ku haboon yahay modelkena)
#    input_arr = np.array([input_arr]) # old convert single image to batch

    predictions = model.predict(input_arr)
    return np.argmax(predictions) #return index of max element

#Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page",["Home","About Us","Disease Detection"])

#Main Page
if(app_mode=="Home"):
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "Home.jpg"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 
    
    
                
    ### How It Works
    
                

    ### Why Choose Us ( The rooters)?
    

    ### Get Started
    
    
    ### About Us

                
    """)

#About Project
elif(app_mode=="About Us"):
    st.header("About Us")
    st.markdown("""
                #### About The Dataset
                

                #### Content
                1. train ( images)
                2. validation ( images)
                3. test ( images)

                """)

#Prediction Page
elif(app_mode=="Disease Detection"):
    st.header("Disease Detection")
    test_image = st.file_uploader("Choose an Image:")
    if(st.button("Show Image")):
        st.image(test_image, width=4, use_container_width=True)
    #Predict button
    if(st.button("Predict")):
        st.snow()
        st.write("Model Report......................................................................................")
        result_index = model_prediction(test_image)
        #Reading Labels
        class_names = ['Blight ', 'Common_Rust', 'Gray_Leaf_Spot', 'Healthy']
        d_name = class_names[result_index]

        if(d_name != "Healthy"):
            st.warning("Model is Predicting it's a {}".format(d_name))
            st.markdown(""" 
            fadlan geedkan waxa haya xanuun, waa in aad sida ugu dhakhsaha badan
            la xidhiidhia  qof ku takhasusay xanuunada dhirta
            """)

        else: 
            st.success("Model is Predicting it's a {}".format(d_name))
            st.markdown(""" 
                Geedkan wuu caafimaad qabaa!

                """)
        
