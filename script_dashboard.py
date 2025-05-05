import streamlit as st
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import requests
import urllib3
import time,json
#########################################
india_timezone = ZoneInfo("Asia/Kolkata")
urllib3.disable_warnings()
#########################################
 

# Set the page title
st.set_page_config(page_title="H&M Infra Operations Automation")


# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "login"  # Start on the login page


st.image("H&M logo.png",)
st.title("H&M Net-Ops")
st.header("Automation Dashboard")
st.write("All your automation tasks at one place")

#page_selection = st.sidebar.radio("Select Page", ["Login", "Dashboard"])
# Login Page
if st.session_state.page == "login":
    st.subheader("Login")
    username = st.text_input("Enter your nofh id:")
    password = st.text_input("Enter your nofh password:",type="password")
    customer_id = st.text_input("Enter your customer id:")
    client_id= st.text_input("Enter your client id:")
    client_secret= st.text_input("Enter your client secret:")
    access_token1= st.text_input("Enter your access token:")
    st.session_state.access_token1 = access_token1

    if st.button("Login"):
        st.success("Credentials saved securely!")
        st.session_state.page = "dashboard"
        #st.session_state.access_token = access_token
        st.rerun()

 
def timenow():
    return datetime.now(india_timezone).strftime('%H:%M:%S-%m/%d/%Y')
 
class status:
    status_df = pd.DataFrame(columns=["Timestamp","Status code","Response"])
    Time=datetime.now(india_timezone).strftime('%H%M%S_%m%d%Y')
    def update(status_code=None,response=None):
        Time=timenow()
        st.write(f"{Time} : {status_code}, {response}")  
        status_list=[Time,status_code,response]
        #status.status_df=pd.concat([status.status_df, pd.DataFrame([status_list], columns=status.status_df.columns)], ignore_index=True)
        #status.status_df.to_excel(f"Migration_status_{status.Time}.xlsx",index=False)
 
class store:
    store_access_token=""
    base_url="https://apigw-eucentral2.central.arubanetworks.com"
    def getToken():
        return store.store_access_token
   
def refreshToken():
    global access_token
    if "access_token1" in st.session_state:
        store.store_access_token = st.session_state.access_token1
        print("Access token updated successfully. 🔃")
#########################################
class reqs:
    def __init__(self,url=None,data=None):
        self.url=f"{store.base_url}{url}"
        self.data=data
        self.total_count=0
        self.offset=0
        self.count=0
   
    @property
    def apiHeaders(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {store.getToken()}"
        }
 
    def get(self):
        while self.count<5:
            try:
                req = requests.get(self.url,headers=self.apiHeaders, verify=False)
                response= req.json()
                if req.status_code != 200:
                    raise Exception("Request failed... ❌")
                status.update(status_code=req.status_code, response=self.data)
                self.count+=1
                return response.get(self.data)
            except Exception as e:
                self.count+=1
                status.update(status_code=req.status_code, response=f"{self.data} : {e}")
                if req.status_code ==500 or req.status_code ==429:
                    time.sleep(10)
                if response.get("error")=='invalid_token':
                    refreshToken()
                elif str(e)==f"'{self.data}'":
                    print(f"Invalid data : {e} 🔄")
   
    def getdf(self):
        while self.count<5:
            try:
                req = requests.get(self.url,headers=self.apiHeaders, verify=False)
                response= req.json()
                if req.status_code != 200:
                    raise Exception("Request failed... ❌")
                if "total" in self.url:
                    self.total_count = int(response["total"])
                status.update(status_code=req.status_code, response=self.data)
                self.df = pd.DataFrame.from_dict(response[self.data])
                self.count+=1
                return self.df
            except Exception as e:
                self.count+=1
                status.update(status_code=req.status_code, response=f"{self.data} : {e}")
                if req.status_code ==500 or req.status_code ==429:
                    time.sleep(10)
                if response.get("error")=='invalid_token':
                    refreshToken()
                elif str(e)==f"'{self.data}'":
                    print(f"Invalid data : {e} 🔄")

 
refreshToken()



st.markdown("""
    <style>
    .big-button {
        width: 100%;
        padding: 20px;
        font-size: 22px;
        font-weight: bold;
        text-align: center;
        color: white;
        border: none;
        border-radius: 12px;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .gateway { background-color: #4CAF50; }
    .switches { background-color: #FF9800; }
    .access_points { background-color: #2196F3; }
    img {
        height: 80px;
        width: 80px;
        margin-right: 20px;
    }
    </style>
""", unsafe_allow_html=True)


if st.session_state.page == "dashboard":
   

    st.title("Dashboard")
    st.write("Welcome to the Automation Dashboard!")
    st.write(" ")

    st.subheader("Select your device for automation:")
    # Create three columns for buttons
    col1, col2, col3 = st.columns(3)

    # Gateway Button
    with col1:
        st.image("Gateway.png", width=100)
        if st.button("🚪 Gateway", key="gateway"):
            
            st.session_state.page = "gateway"
            st.rerun()

    # Switches Button
    with col2:
        st.image("Switch.png", width=100)
        if st.button("🔌 Switches", key="switches"):
            st.session_state.page = "switches"

    # Access Points Button
    with col3:
        st.image("Ap.png", width=100)
        if st.button("📶 Access Points", key="access_points"):
            st.session_state.page = "access_points"

    # Show Selected Section
if st.session_state.page == "gateway":
    st.title("Gateway")
    st.image("Gateway.png")
    st.subheader("Gateway Automation tasks")
    if st.button("Group level Gateway configuration"):
        st.session_state.page = "Group level Gateway configuration"
        st.rerun()

if st.session_state.page == "Group level Gateway configuration":        
        group_name = st.text_input("Enter group name:")
        if st.button("Check Config"):
            st.write(f"Fetching configuration for group: {group_name}")
            config_url = f"/caasapi/v1/showcommand/object/committed?group_name={group_name}"
            get_config = reqs(url=config_url, data="config").getdf()
            st.write(get_config)
            st.success("Configuration fetched successfully!")

elif st.session_state.page == "switches":
    st.subheader("Switches Settings")
    st.write("Fetching Switches Data...")

elif st.session_state.page == "access_points":
    st.subheader("Access Points Settings")
    st.write("Fetching Access Point Data...")