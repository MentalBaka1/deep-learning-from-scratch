"""
第11章·第3节·Agent框架
核心: Tool Use/Function Calling, ReAct循环(Thought→Action→Observation), 多Agent协作, Agent记忆

Agent = LLM + 工具调用 + 规划能力 + 记忆
本节实现一个完整的Agent框架，包含工具定义、ReAct循环和多Agent协作。
"""

import json
import random
from typing import List, Dict, Callable, Optional, Any
from dataclasses import dataclass

# ============================================================
# 第一部分：工具定义与Function Calling
# ============================================================

@dataclass
class ToolDefinition:
    """
    工具定义，遵循OpenAI Function Calling的JSON Schema格式。
    每个工具包含：名称、描述、参数定义、实际执行函数。
    """
    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema格式
    func: Callable

    def to_schema(self) -> Dict:
        """导出为Function Calling格式的JSON Schema"""
        return {"type": "function", "function": {
            "name": self.name, "description": self.description,
            "parameters": self.parameters}}

# ——— 定义具体工具 ———

def calculator(expression: str) -> str:
    """安全计算器：只允许基础数学运算"""
    allowed = set("0123456789+-*/.() ")
    if not all(c in allowed for c in expression):
        return "错误：表达式包含不允许的字符"
    try:
        return f"计算结果：{eval(expression)}"
    except Exception as e:
        return f"计算错误：{e}"

def weather_lookup(city: str) -> str:
    """模拟天气查询"""
    db = {"北京": "晴，25°C", "上海": "多云，28°C",
          "深圳": "雷阵雨，32°C", "杭州": "阴，23°C"}
    return db.get(city, f"未找到{city}的天气信息")

def web_search(query: str) -> str:
    """模拟网络搜索"""
    db = {"Python": "Python是一种高级编程语言，由Guido van Rossum于1991年发布。",
          "Transformer": "Transformer是2017年由Google提出的神经网络架构。",
          "RAG": "RAG(检索增强生成)结合了信息检索和文本生成技术。"}
    for key, val in db.items():
        if key.lower() in query.lower():
            return f"搜索结果：{val}"
    return f"搜索'{query}'：未找到相关结果"

# 创建工具列表
TOOLS = [
    ToolDefinition("calculator", "执行数学计算",
        {"type": "object",
         "properties": {"expression": {"type": "string", "description": "数学表达式"}},
         "required": ["expression"]},
        lambda args: calculator(args["expression"])),
    ToolDefinition("weather", "查询城市天气",
        {"type": "object",
         "properties": {"city": {"type": "string", "description": "城市名称"}},
         "required": ["city"]},
        lambda args: weather_lookup(args["city"])),
    ToolDefinition("search", "搜索网络信息",
        {"type": "object",
         "properties": {"query": {"type": "string", "description": "搜索关键词"}},
         "required": ["query"]},
        lambda args: web_search(args["query"])),
]

print("=" * 60)
print("【Function Calling Schema 示例】")
for tool in TOOLS:
    s = tool.to_schema()["function"]
    print(f"  {s['name']}: {s['description']}")
    print(f"    参数: {json.dumps(s['parameters'], ensure_ascii=False)}")

# ============================================================
# 第二部分：Agent 记忆系统
# ============================================================

@dataclass
class Message:
    """对话消息"""
    role: str            # "user" / "assistant" / "tool" / "system"
    content: str
    tool_name: Optional[str] = None

class AgentMemory:
    """
    Agent记忆管理：短期记忆（对话历史）+ 长期记忆（总结/知识）。
    超过容量时，将旧消息总结后移入长期记忆。
    """
    def __init__(self, max_short_term: int = 20):
        self.short_term: List[Message] = []
        self.long_term: List[str] = []
        self.max_short_term = max_short_term

    def add(self, msg: Message):
        self.short_term.append(msg)
        if len(self.short_term) > self.max_short_term:
            old = self.short_term[:5]
            self.long_term.append(f"早期对话摘要：{len(old)}轮内容")
            self.short_term = self.short_term[5:]

    def get_context(self) -> str:
        parts = []
        if self.long_term:
            parts.append("【长期记忆】\n" + "\n".join(self.long_term))
        parts.append("【对话历史】")
        for m in self.short_term:
            parts.append(f"[{m.role}] {m.content}")
        return "\n".join(parts)

# ============================================================
# 第三部分：ReAct Agent 核心实现
# ============================================================

class ReActAgent:
    """
    ReAct (Reasoning + Acting) Agent。
    执行循环：Thought → Action → Observation → ... → Final Answer

    相比纯LLM：可调用外部工具、推理过程透明、减少幻觉。
    """
    def __init__(self, tools: List[ToolDefinition], max_steps: int = 5):
        self.tools = {t.name: t for t in tools}
        self.max_steps = max_steps
        self.memory = AgentMemory()

    def _mock_think(self, question: str, step: int) -> Dict:
        """
        模拟LLM思考过程。实际应用中调用LLM API。
        根据问题关键词选择工具（模拟LLM推理）。
        """
        if step == 1:
            if "天气" in question:
                city = "北京"
                for c in ["上海", "深圳", "杭州"]:
                    if c in question:
                        city = c
                return {"thought": f"需要查询{city}天气",
                        "action": "weather", "action_input": {"city": city}}
            elif any(k in question for k in ["计算", "加", "乘", "+", "*"]):
                return {"thought": "需要数学计算",
                        "action": "calculator", "action_input": {"expression": "42 * 2"}}
            else:
                return {"thought": "需要搜索相关信息",
                        "action": "search", "action_input": {"query": question}}
        return {"thought": "已获得足够信息", "final_answer": "根据查询结果，已找到答案。"}

    def run(self, question: str) -> str:
        """执行ReAct循环处理用户问题"""
        print(f"\n  {'─'*50}")
        print(f"  用户问题：{question}")
        self.memory.add(Message("user", question))

        for step in range(1, self.max_steps + 1):
            decision = self._mock_think(question, step)
            thought = decision.get("thought", "")
            print(f"  [Thought {step}] {thought}")

            # 最终答案
            if "final_answer" in decision:
                answer = decision["final_answer"]
                print(f"  [Final] {answer}")
                self.memory.add(Message("assistant", answer))
                return answer

            # 执行工具
            action_name = decision["action"]
            action_input = decision["action_input"]
            print(f"  [Action {step}] {action_name}({json.dumps(action_input, ensure_ascii=False)})")
            observation = (self.tools[action_name].func(action_input)
                          if action_name in self.tools else f"未知工具'{action_name}'")
            print(f"  [Observation {step}] {observation}")
            self.memory.add(Message("tool", observation, action_name))

        return "达到最大推理步数"

print("\n" + "=" * 60)
print("【ReAct Agent 演示】")
agent = ReActAgent(TOOLS)
agent.run("北京今天天气怎么样？")
agent.run("帮我计算 42 乘以 2")
agent.run("什么是Transformer？")

# ============================================================
# 第四部分：多Agent协作
# ============================================================

class SpecializedAgent:
    """专用Agent：只擅长处理特定类型的任务"""
    def __init__(self, name: str, expertise: str, keywords: List[str]):
        self.name = name
        self.expertise = expertise
        self.keywords = keywords

    def can_handle(self, query: str) -> float:
        """评估能否处理该查询，返回置信度(0-1)"""
        score = sum(1 for kw in self.keywords if kw in query.lower())
        return min(score / max(len(self.keywords), 1), 1.0)

    def process(self, query: str) -> str:
        return f"[{self.name}] 基于{self.expertise}已处理：{query[:30]}..."

class OrchestratorAgent:
    """
    编排Agent：将任务分配给最合适的专用Agent。
    多Agent协作模式：
      1. Router模式 —— 根据任务类型路由（本例）
      2. 层级模式 —— Manager分解任务，Worker执行子任务
      3. 辩论模式 —— 多Agent对同一问题给出观点后综合
    """
    def __init__(self, agents: List[SpecializedAgent]):
        self.agents = agents

    def route(self, query: str) -> str:
        print(f"\n  [编排器] 查询：{query}")
        scores = [(ag, ag.can_handle(query)) for ag in self.agents]
        for ag, sc in scores:
            print(f"    {ag.name} 匹配度：{sc:.2f}")
        scores.sort(key=lambda x: x[1], reverse=True)
        best, best_score = scores[0]
        if best_score > 0:
            print(f"  [编排器] → {best.name}")
            return best.process(query)
        return f"无合适Agent处理：{query[:30]}..."

agents = [
    SpecializedAgent("数学Agent", "数学计算", ["计算", "求解", "方程", "加", "乘"]),
    SpecializedAgent("天气Agent", "气象信息", ["天气", "气温", "下雨", "预报"]),
    SpecializedAgent("代码Agent", "编程开发", ["代码", "编程", "python", "函数"]),
    SpecializedAgent("知识Agent", "百科问答", ["什么是", "介绍", "历史", "原理"]),
]
orchestrator = OrchestratorAgent(agents)

print("\n" + "=" * 60)
print("【多Agent协作 — Router模式】")
for q in ["帮我计算123+456", "明天北京天气预报",
           "用Python写排序函数", "什么是量子计算？"]:
    result = orchestrator.route(q)
    print(f"  结果：{result}\n")

# ============================================================
# 第五部分：总结 & 思考题
# ============================================================

print("=" * 60)
print("""
【Agent 框架核心概念总结】
  组成：LLM（大脑）+ 工具（手脚）+ 记忆（经验）+ 规划（策略）
  关键模式：
    1. ReAct：Thought→Action→Observation循环
    2. Function Calling：结构化工具调用（JSON Schema）
    3. 多Agent协作：Router / 层级 / 辩论
    4. 记忆管理：短期（对话）+ 长期（总结/向量存储）
  主流框架：LangChain, LlamaIndex, AutoGPT, CrewAI, AutoGen

【思考题】
  1. ReAct框架中，如何防止Agent陷入无限循环？除了最大步数还有哪些策略？

  2. Function Calling的JSON Schema设计有什么要点？
     如何让LLM更准确地选择和调用工具？

  3. 多Agent协作中如何处理冲突？Router模式和层级模式各适用什么场景？

  4. Agent的长期记忆应存储哪些信息？如何平衡记忆容量和检索效率？

  5. 对比LangChain和AutoGen的设计理念，它们在Agent实现上有什么异同？
""")

if __name__ == "__main__":
    print("本节演示完毕。将_mock_think替换为真实LLM API即可用于生产。")
