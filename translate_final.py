import re

files = [
    "frontend/app/insights/page.jsx",
    "frontend/app/agent/page.jsx",
    "frontend/app/components/ForecastClient.jsx",
    "frontend/app/components/AgentClient.jsx",
    "frontend/app/report/page.jsx",
]

translations = {
    "历史实际": "Historical Actual",
    "最佳模型": "Best Model",
    "拟合": "Fitted",
    "可查看": "can view",
    "并自动": "and auto",
    "用于": "used for",
    "撰写": "writing",
    "例如": "e.g.",
    "该数据为某零食在华东地区的每": "daily data for snacks in East China",
    "销售金额与销量": "sales amount and volume",
    "降低缺货并控制库存": "reduce stockouts and control inventory",
    "期数": "Periods",
    "上传文件后点击": "Click after uploading",
    "系统会展示": "system will show",
    "并自动": "and auto",
    "预览中": "Previewing",
    "没有数据": "No data",
    "正在执行": "Executing",
    "图表": "Chart",
    "结果": "Results",
    "运行": "Run",
    "返回": "Return",
    "提交": "Submit",
    "完成": "Complete",
    "失败": "Failed",
    "错误": "Error",
}

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    for k, v in translations.items():
        content = content.replace(k, v)
        
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)
    print(f"Translated: {f}")

