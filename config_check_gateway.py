from netmiko import ConnectHandler
import paramiko
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
                return self.df
            except Exception as e:
                if response.get("error")=='invalid_token':
                    refreshToken()
                elif str(e)==f"'{self.data}'":
                    print(f"Invalid data : {e} ")
                print(f"{e} : {response}\nRetrying...🔄")
 
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
    
    def text(self):
        while True:
            try:
                req = requests.get(self.url,headers=self.apiHeaders, verify=False)
                response= req.text  
               
                if req.status_code != 200:
                    raise Exception("Request failed... ❌")
                return response
            except Exception as e:
                response=req.json()
                if response.get("error")=='invalid_token':
                    refreshToken()
                elif str(e)==f"'{self.data}'":
                    print(f"Invalid data : {e} 🔄")


access_token = input("Enter the access token: ")
store.store_access_token=access_token
#groups=input("Enter the group template: ")




"""allTemplate_url = f"/configuration/v1/groups/{groups}/templates?limit=20&offset=0"
allTemplate_df = reqs(allTemplate_url,"data").getdf()
Template_name = list(allTemplate_df["name"])"""


template_df = pd.read_excel("template_input.xlsx")
template_list = list(template_df["Template"])
keyword = input("Enter the keyword: ").split(",")


columns = ["Template_name",f"{keyword}"]
output_df = pd.DataFrame(columns = columns)
for i in template_list:
    gateway_text_url = f"/caasapi/v1/showcommand/object/committed?group_name={i}"
    template_text = reqs(url = gateway_text_url).text()
    print(template_text)
    output=[]
    output.append(i)
    count = 0
    for j in keyword:
        if j in template_text:
            count +=1
            output.append(count)
    if count==2:
        count="Yes"
    else:
        count="No"    
    output_df = pd.concat([output_df,pd.DataFrame({"Template_name":[i],f"{keyword}":[count]})],ignore_index=True)
output_df.to_excel("Keyword_Result.xlsx",index = False)



