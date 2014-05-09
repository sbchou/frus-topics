
# Perform Topic Modelling on predicted redacted text using Text Baslines of Document Alignment model.

# In[3]:

get_ipython().magic(u'load_ext autoreload')
get_ipython().magic(u'autoreload 2')


# In[1]:

import redact.baselines as baselines
import redact.experiments
from redact.data.human import *
import redact.data.passwd as passwd


# In[2]:

db = passwd.get_db()
cursor = passwd.get_cursor(db)
all = HumanPair.get_all(cursor)


# Find the most common topics of the predicted redacted text.

# In[4]:

IMAGE_PATH = "/media/sophie/Sophie Drive/images/Img/"
documents = []
failed = []

for i, a in enumerate(all):
    try:
        documents.append(baselines.make_docs(a, IMAGE_PATH))  
    except Exception:
        print "fail", i
        failed.append(i)
        pass


# Out[4]:

#     fail 123
# 

# Now make the predictions, using the baseline Text Aligner.

# In[5]:

text_aligner = baselines.TextAligner()
predictions = [text_aligner.align(p1, p2, i)
                   for i, (p1, p2) in enumerate(documents)]
# Each predicted redaction text will be 
# treated as a doc in the topic model
predictions = [list(p) for p in predictions]
predicted_text = [p.text for doc in predictions for p in doc]


# Now for the Topic Modelling:

# In[7]:

import topicmodel.topicmodel as topicmodel
m = topicmodel.TopicModel(predicted_text)


# In[11]:

lda = m.lda(20)
for t in lda.show_topics(topics=-1, topn=20):
    print t
    print


# Out[11]:

#     WARNING:gensim.models.ldamodel:too few updates, training might not converge; consider increasing the number of passes or iterations to improve accuracy
# 

#     0.015*view + 0.013*told + 0.013*however + 0.013*n + 0.013*congress + 0.013*make + 0.013*d + 0.013*quang + 0.011*point + 0.009*australians + 0.009*years + 0.008*public + 0.007*position + 0.007*within + 0.007*confusion + 0.007*f + 0.007*possibility + 0.007*serious + 0.007*gvn + 0.007*result
#     
#     0.029*refugees + 0.024*committee + 0.019*nuclear + 0.017*united + 0.017*states + 0.014*problem + 0.013*capabilities + 0.013*army + 0.012*assist + 0.011*solution + 0.010*british + 0.010*period + 0.008*recommend + 0.008*probably + 0.008*time + 0.007*far + 0.007*trying + 0.007*continuing + 0.006*page + 0.006*election
#     
#     0.020*secretary + 0.017*actions + 0.016*british + 0.013*made + 0.010*ea + 0.010*felt + 0.010*give + 0.009*deal + 0.009*commission + 0.009*specific + 0.008*use + 0.008*response + 0.007*end + 0.007*idea + 0.007*necessary + 0.007*planning + 0.007*clearly + 0.007*therefore + 0.007*taken + 0.006*might
#     
#     0.024*president + 0.017*british + 0.012*united + 0.011*may + 0.011*mccone + 0.010*asked + 0.009*states + 0.008*french + 0.008*intelligence + 0.008*ea + 0.008*labor + 0.007*state + 0.007*thought + 0.007*soviet + 0.007*support + 0.006*position + 0.006*available + 0.006*future + 0.006*defense + 0.006*another
#     
#     0.021*part + 0.021*president + 0.017*military + 0.016*m + 0.011*secretary + 0.010*united + 0.009*force + 0.009*subject + 0.009*recognized + 0.009*april + 0.009*situation + 0.009*government + 0.009*states + 0.009*british + 0.008*c + 0.007*political + 0.007*done + 0.007*dulles + 0.007*study + 0.007*meeting
#     
#     0.019*soviet + 0.018*say + 0.014*e + 0.013*british + 0.013*know + 0.013*twice + 0.013*french + 0.010*position + 0.010*bloc + 0.010*radio + 0.010*possibility + 0.010*among + 0.009*believe + 0.009*within + 0.008*c + 0.007*get + 0.007*whether + 0.007*use + 0.007*meeting + 0.007*objectives
#     
#     0.016*weapons + 0.016*yield + 0.016*put + 0.016*inr + 0.014*stated + 0.011*think + 0.011*telegram + 0.011*respect + 0.011*step + 0.011*zone + 0.011*phase + 0.010*likely + 0.010*defense + 0.009*secretary + 0.009*nasser + 0.008*area + 0.008*state + 0.008*position + 0.007*staff + 0.007*allen
#     
#     0.021*military + 0.019*turkish + 0.017*effort + 0.016*force + 0.014*dr + 0.014*intelligence + 0.011*support + 0.011*continue + 0.010*government + 0.010*levels + 0.010*japan + 0.010*kissinger + 0.010*slow + 0.010*observed + 0.010*making + 0.010*national + 0.010*director + 0.009*program + 0.009*british + 0.008*review
#     
#     0.038*z + 0.024*e + 0.023*state + 0.022*page + 0.019*fm + 0.015*house + 0.015*white + 0.015*de + 0.013*c + 0.012*defense + 0.012*cia + 0.012*dia + 0.012*j + 0.012*may + 0.012*r + 0.012*p + 0.011*war + 0.008*military + 0.008*soviet + 0.008*present
#     
#     0.026*government + 0.014*one + 0.013*united + 0.013*states + 0.013*countries + 0.013*sent + 0.011*saragat + 0.011*geneva + 0.011*respect + 0.010*laos + 0.010*say + 0.009*conversations + 0.009*secretary + 0.008*british + 0.008*may + 0.008*way + 0.007*prime + 0.007*foreign + 0.007*minister + 0.006*military
#     
#     0.029*secretary + 0.019*situation + 0.017*government + 0.012*present + 0.012*position + 0.011*take + 0.010*united + 0.010*president + 0.010*states + 0.009*action + 0.009*iraq + 0.008*countries + 0.008*dulles + 0.008*humphrey + 0.008*sure + 0.008*committed + 0.008*embassy + 0.008*control + 0.008*secret + 0.007*soviet
#     
#     0.038*dulles + 0.024*china + 0.023*secretary + 0.012*identified + 0.012*chinese + 0.012*overseas + 0.010*british + 0.010*man + 0.010*president + 0.009*allen + 0.009*council + 0.008*situation + 0.008*government + 0.008*foreign + 0.007*security + 0.007*whether + 0.007*study + 0.007*oil + 0.006*day + 0.006*communist
#     
#     0.016*soviet + 0.014*illegible + 0.013*text + 0.013*secretary + 0.011*believe + 0.011*policy + 0.011*foreign + 0.011*vietnam + 0.011*leaders + 0.011*iranian + 0.009*forces + 0.008*combat + 0.008*came + 0.008*serious + 0.008*decision + 0.008*station + 0.008*reactions + 0.008*soviets + 0.008*soil + 0.006*political
#     
#     0.019*forces + 0.014*say + 0.013*soviet + 0.011*situation + 0.010*state + 0.009*president + 0.008*drv + 0.008*united + 0.008*problem + 0.008*government + 0.007*action + 0.007*may + 0.007*make + 0.007*see + 0.007*current + 0.007*time + 0.007*text + 0.007*mccone + 0.007*amembassy + 0.007*difficulties
#     
#     0.021*states + 0.018*united + 0.016*communist + 0.016*party + 0.014*may + 0.013*special + 0.013*american + 0.010*support + 0.010*president + 0.008*latin + 0.008*stated + 0.008*part + 0.007*committee + 0.007*outlook + 0.007*dated + 0.007*iran + 0.007*prospects + 0.007*soviet + 0.007*discussion + 0.007*interest
#     
#     0.021*situation + 0.013*drv + 0.013*base + 0.013*new + 0.013*position + 0.010*decline + 0.010*whole + 0.010*security + 0.010*maintain + 0.010*world + 0.009*power + 0.009*major + 0.008*area + 0.008*continue + 0.008*atomic + 0.008*united + 0.007*nato + 0.007*three + 0.007*advanced + 0.007*added
#     
#     0.060*text + 0.060*illegible + 0.019*states + 0.017*united + 0.017*shah + 0.017*one + 0.012*president + 0.010*power + 0.010*e + 0.010*position + 0.010*n + 0.010*nato + 0.007*military + 0.007*however + 0.007*something + 0.007*situation + 0.007*money + 0.007*iranian + 0.007*people + 0.007*iran
#     
#     0.020*johnson + 0.014*radio + 0.014*possible + 0.014*courses + 0.012*years + 0.012*prepare + 0.012*get + 0.012*commission + 0.012*action + 0.011*gvn + 0.010*possibility + 0.010*order + 0.010*day + 0.009*m + 0.009*directing + 0.008*facilities + 0.007*one + 0.007*report + 0.007*various + 0.007*attached
#     
#     0.013*secretary + 0.011*war + 0.011*president + 0.011*military + 0.010*course + 0.010*take + 0.010*defense + 0.009*dulles + 0.009*probably + 0.009*point + 0.009*give + 0.008*united + 0.008*however + 0.008*several + 0.007*felt + 0.007*case + 0.007*one + 0.007*much + 0.007*rapid + 0.007*russian
#     
#     0.021*military + 0.019*atomic + 0.016*weapons + 0.013*secret + 0.012*page + 0.012*special + 0.012*national + 0.011*committee + 0.011*policies + 0.010*united + 0.010*general + 0.009*stassen + 0.009*nuclear + 0.009*cooperation + 0.009*uk + 0.009*level + 0.009*basis + 0.009*covert + 0.009*among + 0.009*control
#     
# 

# In[ ]:



