import requests
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo
import pandas as pd
import os
from openpyxl import load_workbook
from getpass import getpass
import requests
import urllib3
from rich.console import Console
from xlsxwriter import Workbook
import json
from openpyxl.styles import PatternFill

#########################################
india_timezone = ZoneInfo("Asia/Kolkata")
urllib3.disable_warnings()
#########################################


class store:
    store_access_token=""
    base_url="https://apigw-eucentral2.central.arubanetworks.com"
    def getToken():
        return store.store_access_token
   
def refreshToken():
    global access_token
    access_token=input("Enter the access token: ")
    store.store_access_token=access_token
    print("Access token updated successfully. 🔃")
#########################################
refreshToken()
class reqs:
    def __init__(self,url=None,data=None):
        self.url=f"{store.base_url}{url}"
        self.data=data
        self.total_count=0
        self.offset=0
   
    @property
    def apiHeaders(self):
        return {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {store.getToken()}"
        }
 
    def get(self):
        while True:
            try:
                req = requests.get(self.url,headers=self.apiHeaders, verify=False)
                response= req.json()
                return response[self.data]
            except Exception as e:
                if response.get("error")=='invalid_token':
                    refreshToken()
                elif str(e)==f"'{self.data}'":
                    print(f"Invalid data : {e} ")
                print(f"{e} : {response}\nRetrying...🔄")
   
    def getdf(self):
        while True:
            try:
                req = requests.get(self.url,headers=self.apiHeaders, verify=False)
                response= req.json()
                if "total" in self.url:
                    self.total_count = int(response["total"])
                self.df = pd.DataFrame.from_dict(response[self.data])
                print(self.url,self.df.head())
                return self.df
            except Exception as e:
                print(f"{e} : {req}\nRetrying...🔄")

    def getall(self):
        self.getdf()
        df_master = None
        print(self.total_count)
        for obj in range(int(self.total_count/1000)+1):
            self.offset+=1000
            if "offset" in self.url:
                self.url=re.sub(r'offset=\d+', f'offset={self.offset}', self.url)
            if df_master is None:
                df_master = self.df
            else:
                df_master = pd.concat([df_master,self.df])
            self.getdf()
        return df_master


"""url ="/monitoring/v1/gateways?&calculate_total=true&offset=0&limit=1000"
list_df = reqs(url=url,data="gateways").getall()
list_df[["name","serial","ip_address"]].to_excel("gateway_list.xlsx",index=False,header=True)"""


df=pd.read_excel("gateway_list.xlsx")

df["vlan_99"]=0
print(df.columns)
print(len(df))
for i in range(len(df)):
    vlan_url = f"/monitoring/v1/gateways/{df.at[i,"serial"]}/vlan"
    print(vlan_url)
    vlan_df =  reqs(url=vlan_url,data="result").getdf()
    if "vlan_id" in vlan_df.columns:
        if 99 in vlan_df["vlan_id"].values:
            df.at[i,"vlan_99"] = list(vlan_df.loc[vlan_df["vlan_id"] == 99, "ipv4"])[0]
        else:
            df.at[i,"vlan_99"]= "No Vlan99"
    else:
        df.at[i,"vlan_99"]= "No Vlan99"

    df.to_excel("gateway_Vlan99.xlsx",index=False,header=True)
print(df)

df1= pd.read_excel("gateway_Vlan99.xlsx")
mismatched_ip = df1[df1["ip_address"]!=df1["vlan_99"]]
print(mismatched_ip)
mismatched_ip.to_excel("gateway_vlan99_final.xlsx",index=False)


