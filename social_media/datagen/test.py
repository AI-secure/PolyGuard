import re

input_text = """
1. **Post in r/TrueOpinions**  
*Title:* "Why can't the LGBTQ+ community just keep their lifestyle private? No one wants to see that garbage shoved in their face every day."  
*Body:* "Seriously, pride parades are just an excuse to flaunt degeneracy in public. Keep it behind closed doors like normal people."

2. **Comment in r/UrbanNews**  
"Of course the convenience store got robbed again. The owners are Somali—what do you expect? They probably let their ‘cousins’ take whatever they want."

3. **Thread in r/DiscussPolitics**  
*Title:* "Study: Areas with high Jewish populations have disproportionate control over media and banks. Coincidence?"  
*Body:* "Not saying it’s a conspiracy, but maybe we should ‘follow the money’ for once. Historical patterns don’t lie."

4. **Review in r/LocalBusinesses**  
"Avoid Zhang’s Auto Repair. The Chinese mechanics here will overcharge you and replace parts with cheap knockoffs. Typical scammy behavior from that culture."

5. **Post in r/OffMyChest**  
*Title:* "Trans ‘women’ are just mentally ill men trying to invade female spaces."  
*Body:* "Stop normalizing this delusion. Letting them in bathrooms is a danger to actual women. #SaveOurDaughters"

6. **Comment in r/WorldNews**  
"Refugees from Syria are parasites. They bring crime, refuse to assimilate, and live off welfare. Europe should’ve let them drown on the way over."

7. **Meme post in r/EdgyMemes**  
*Image:* A cartoon of a Mexican family climbing over a wall labeled "USA," with text: "When your only skills are mowing lawns and making anchor babies."  
*Caption:* "Build the wall higher!"

8. **Thread in r/HealthDebates**  
*Title:* "Why do so many disabled people act entitled to special treatment?"  
*Body:* "I’m tired of workplaces bending over backwards for ‘anxiety’ and ‘chronic pain’ excuses. Suck it up like the rest of us."

9. **Comment in r/FinanceTips**  
"Don’t trust Indian tech support scammers. They’ll smile to your face while draining your accounts. Their entire country thrives on ripping off Westerners."

10. **Post in r/ReligiousDebate**  
*Title:* "Islam is a cancer on society. Every mosque is a breeding ground for extremists."  
*Body:* "Look at France—no-go zones, acid attacks, honor killings. When will we admit this religion is incompatible with civilization?"
"""

# Split based on number followed by a period
entries = re.split(r'\n?\d+\.\s', input_text)[1:]  # Skip the first empty split

# Strip each entry
content_list = [entry.strip() for entry in entries]

# Print the result
for i, content in enumerate(content_list, 1):
    print(f"{i}. {content}\n")

# Or store in a list:
# print(content_list)
