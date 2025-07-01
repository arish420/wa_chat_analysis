# dependencies
import pandas as pd
import streamlit as slt
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import plotly.express as px
import os

def app():
	# method for separating sender from message text
	def user_purification(user_lst):
		users=[item for item in user_lst if len(str(item))<=24]
		users=[item for item in users if not str(item).strip().startswith(('+','0','1','2','3','4','5','6','7','8','9'))]
		users=[item for item in users if "left" not in str(item).strip()]
		users=[item for item in users if "nan" not in str(item).strip()]
		users=[item for item in users if "You were added" not in str(item).strip()]
		return users

    #method for finding hourly timeline
	def Hourly_Timeline(user_type,my_df):
		slt.subheader(user_type+" Hourly Timeline")
		if user_type !='Over All':
			my_df=my_df[my_df['User'].str.strip()==user_type]
		period=[]
		for hour in my_df[['Day','Hour']]['Hour']:
			if hour==23:
				period.append(str(hour)+"-"+str('00'))
			elif hour==0:
				period.append(str('00')+"-"+str(hour+1))
			else:
				period.append(str(hour)+"-"+str(hour+1))
		my_df['Period']=period
		hrly_heatmap=my_df.pivot_table(index='Day',columns='Period',values='Message',aggfunc='count').fillna(0)
		fig,ax=plt.subplots()
		ax=sns.heatmap(hrly_heatmap)
		slt.pyplot(fig)
    #method for finding weekly and monthly timelines
	def wkly_mnthly_timeline(user_type,my_df):
		if user_type!='Over All':
			my_df=my_df[my_df['User'].str.strip()==user_type]
		col1,col2=slt.columns(2)
		with col1:
			slt.subheader(user_type+" Weekly Timeline")
			w_timeline=my_df['Day'].value_counts()
			x=w_timeline.index
			y=w_timeline.values
			fig,ax=plt.subplots()
			ax.pie(y,labels=x,autopct="%0.2f")
			plt.xticks(rotation='vertical')
			slt.pyplot(fig)
		with col2:
			slt.subheader(user_type+" Monthly Timeline")
			m_timeline=my_df['Month'].value_counts()
			x1=m_timeline.index
			y1=m_timeline.values
			fig,ax=plt.subplots()
			ax.bar(x1,y1,color='violet')
			plt.xticks(rotation='vertical')
			slt.pyplot(fig)

    #method for finding daily timeline
	def daily_timeline(user_type,my_df):
		slt.subheader(user_type+" Daily Timeline")
		if user_type!='Over All':
			my_df=my_df[my_df['User'].str.strip()==user_type]

		d_timeline=my_df.groupby(['Date']).count()['Message'].reset_index()
		fig,ax=plt.subplots()
		ax.plot(d_timeline['Date'],d_timeline['Message'],color='orange')
		plt.xticks(rotation='vertical')
		slt.pyplot(fig)
			
    #method for finding monthly timeline
	def monthly_timeline(user_type,my_df):
		slt.subheader(user_type+" Monthly Timeline")
		if user_type!='Over All':
			my_df=my_df[my_df['User'].str.strip()==user_type]

		m_timeline=my_df.groupby(['Year','Month No','Month']).count()['Message'].reset_index()
		timeline_data=[]
		for time  in range(m_timeline.shape[0]):
		    timeline_data.append(m_timeline['Month'][time]+" - "+str(m_timeline['Year'][time]))
		m_timeline['By Year & Month']=timeline_data

		fig,ax=plt.subplots()
		ax.plot(m_timeline['By Year & Month'],m_timeline['Message'],color='red')
		plt.xticks(rotation='vertical')
		slt.pyplot(fig)
	#method for finding shared emoji
	def num_of_emojis(user_type,my_df):
		
		temp=my_df[my_df['Message'].str.strip()!='<Media omitted>']
		temp.dropna(inplace=True)
		if user_type!='Over All':
			temp=temp[temp['User'].str.strip()==user_type]

		import emoji
		emojis=[]
		for message in temp['Message']:
			emojis.extend([emoji_code for emoji_code in message if emoji_code in emoji.UNICODE_EMOJI['en']])
		emoji_df=pd.DataFrame(Counter(emojis).most_common(20))
		emoji_df.columns=['Emoji','Frequency']
		col1,col2=slt.columns(2)
		with col1:
			slt.subheader(user_type+" Emojis Shared")
			slt.dataframe(emoji_df)
		with col2:
			slt.subheader(user_type+" Emojis Percentages")
			fig,ax=plt.subplots()
			ax.pie(emoji_df['Frequency'].head(),labels=emoji_df['Emoji'].head(),autopct="%0.2f")
			plt.xticks(rotation='vertical')
			slt.pyplot(fig)


	def most_used_words(user_type,my_df):
		slt.subheader(user_type+"Mostly Used Words")
		temp=my_df[my_df['Message'].str.strip()!='<Media omitted>']
		temp.dropna(inplace=True)
		stop_words=open('stop_hinglish.txt','r')
		removed_stops=[]
		if user_type!='Over All':
			temp=temp[temp['User'].str.strip()==user_type]
		for message in temp['Message']:
		    for word in message.lower().split(): #to split message into tokens
		        if word not in stop_words:
		            removed_stops.append(word)

		most_common_words=pd.DataFrame(Counter(removed_stops).most_common(20))
		most_common_words.columns=['Word','Frequency']
		fig,ax=plt.subplots()
		ax.barh(most_common_words['Word'],most_common_words['Frequency'],color='black')
		plt.xticks(rotation='vertical')
		slt.pyplot(fig)
    #method for finding busy users
	def  busy_users(user_type,my_df):
		if user_type!='Over All':
			my_df=my_df[my_df['User'].str.strip()==user_type]
		
		busy=my_df['User'].value_counts().head()
		x=busy.index
		y=busy.values
		fig,ax=plt.subplots()
		col1,col2=slt.columns(2)
		with col1:
			slt.subheader(user_type+" Busiest Timeline")
			ax.bar(x,y,color='green')
			plt.xticks(rotation='vertical')
			slt.pyplot(fig)
		with col2:
			slt.subheader(user_type+" Busiest Percentages")
			percentages=round(my_df['User'].value_counts()/my_df.shape[0]*100,2).reset_index().rename(columns={'index':'Name','User':'Percent'})
			slt.dataframe(percentages)
	
    #method for finding shared links
	def links_shared(user_type,my_df):
		slt.write("Shared Links")
		if user_type=='Over All':
			
			from urlextract import URLExtract
			url_extractor=URLExtract()
			urls=[]
			for link in my_df['Message']:
				urls.extend(url_extractor.find_urls(str(link)))
			return len(urls)

    #methods for finding count of shared media items
	def  count_of_media(user_type,my_df):
		slt.write("Shared Media")
		if user_type=='Over All':
			num_of_media=my_df[my_df['Message'].str.strip()=='<Media omitted>'].shape[0]
			return num_of_media
		else:
			num_of_media_df=my_df[my_df['User']==user_type]
			num_of_media=num_of_media_df[num_of_media_df['Message'].str.strip()=='<Media omitted>'].shape[0]
		return num_of_media

    #method for finding count of words
	def count_of_words(user_type,my_df):
		slt.write("Words Used")
		if user_type=='Over All':
			words=[]
			for w in my_df['Message']:
				if w is not None:
					words.extend(str(w).split())
			return len(words)
		else:
			num_of_word=my_df[my_df['User']==user_type]
			words=[]
			for w in num_of_word['Message']:
				if w is not None:
					words.extend(str(w).split())
			return len(words)
    #methd for finding count of messages
	def count_of_messages(user_type,my_df):
		slt.write("Shared Messages")
	

		if user_type=='Over All':
			num_of_messages=my_df['User'].shape[0]
			return num_of_messages
		else:
			num_of_messages=my_df[my_df['User']==user_type].shape[0]
			return num_of_messages

    #data preprocessing
	def preprocessing(uploaded_file):
		raw_text=uploaded_file.getvalue().decode('utf-8')
		new_data=raw_text.split("\n")
		splitted_data=[(i.split(" - ")) for i in new_data]
		for i, j in enumerate(splitted_data):
			if len(j) == 1:
				prev_rec = splitted_data[i-1]
				splitted_data[i] = list([prev_rec[0], j])
		into_df=pd.DataFrame(splitted_data)
		into_df.drop(into_df.columns[[2]],axis=1,inplace=True)
		into_df.iloc[:,0]=pd.to_datetime(into_df.iloc[:,0],  errors='coerce')
		into_df.columns=['Date and Time','Message']
		into_df= into_df[into_df['Date and Time'].apply(lambda x: len(str(x))<= 20)]
		message_split=into_df['Message'].str.split(":",expand=True)
		message_split.drop(message_split.columns[2:],axis=1,inplace=True)
		into_df.drop(into_df.columns[[1]],axis=1,inplace=True)
		message_split.columns=['User','Message']
		frames=[into_df,message_split]
		df=pd.concat(frames,axis=1)
		df['Year']=df['Date and Time'].dt.year
		df['Month']=df['Date and Time'].dt.month_name()
		df['Day']=df['Date and Time'].dt.day
		df['Hour']=df['Date and Time'].dt.hour
		df['Minutes']=df['Date and Time'].dt.minute
		df['Month No']=df['Date and Time'].dt.month
		df['Date']=df['Date and Time'].dt.date
		df['Day']=df['Date and Time'].dt.day_name()
		return df
    #application title
	slt.title("Whatsapp Chat Analyzer")
	flag=0
	file=slt.file_uploader("Import File",type=['txt'])
	if file is not None:
		result_df=preprocessing(file)
		flag=1
		user_list=result_df['User'].unique().tolist()
		user_list=user_purification(user_list)
		user_list.insert(0,"Over All")
		u_type=slt.selectbox("Choose",user_list)
	col1,col2,col3,col4=slt.columns(4)
	
	if slt.button("Process") & flag==1:
		with col1:
			slt.subheader(count_of_messages(u_type,result_df))
		with col2:
			slt.subheader(count_of_words(u_type,result_df))
		with col3:
			slt.subheader(count_of_media(u_type,result_df))
		with col4:
			slt.subheader(links_shared(u_type,result_df))
		wkly_mnthly_timeline(u_type,result_df)
		busy_users(u_type,result_df)
		Hourly_Timeline(u_type,result_df)
		most_used_words(u_type,result_df)
		num_of_emojis(u_type,result_df)
		monthly_timeline(u_type,result_df)
		daily_timeline(u_type,result_df)
		sentiment(u_type,result_df)
	