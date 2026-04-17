import re
import glob

files = [
    "frontend/app/forecast/page.jsx",
    "frontend/app/insights/page.jsx",
    "frontend/app/agent/page.jsx",
    "frontend/app/components/ForecastClient.jsx",
    "frontend/app/components/AgentClient.jsx",
    "frontend/app/preview/page.jsx",
    "frontend/app/report/page.jsx",
    "frontend/app/upload/page.jsx",
    "frontend/app/page.jsx",
]

matches = set()
for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
        found = re.findall(r'[\u4e00-\u9fff][\u4e00-\u9fff，、。？！（）\s0-9A-Za-z\/\-_]+[\u4e00-\u9fff]', content)
        for m in found:
            matches.add(m)
        
        # also find isolated single chars or small bits
        found2 = re.findall(r'"[^"]*[\u4e00-\u9fff]+[^"]*"', content)
        for m in found2:
            matches.add(m.strip('"'))
            
        found3 = re.findall(r'>([^<]*[\u4e00-\u9fff]+[^<]*)<', content)
        for m in found3:
            matches.add(m.strip())

for m in sorted(list(matches)):
    print(repr(m))

