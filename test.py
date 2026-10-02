#from tools.tavily_tool import tavily_search
#from tools.flight_tool import search_flights

#res= tavily_search("Best hotels in india")
#print(res)

#res = search_flights("plan a 21 day trip from JFK,United states to Delhi,india")
#print(res)

import sys
sys.stdout.reconfigure(encoding="utf-8")  # Windows console defaults to cp1252

from backend import review_travel_plan, run_travel_agent

# Quick tool tests (uncomment to run one at a time)
# from tools.tavily_tool import tavily_search
# print(tavily_search("Best hotels in Goa"))
#
# from tools.flight_tool import search_flights
# print(search_flights("Plan a 4 days trip from Delhi to Goa"))

user_input = input("Enter travel request: ")

response = run_travel_agent(
    user_input=user_input,
    thread_id="test_user",
)

# Human review: the plan waits until you approve it (or ask for changes)
while response["status"] == "awaiting_review":
    print("\nDRAFT PLAN:\n")
    print(response["answer"])
    changes = input("\nPress Enter to approve, or type what should change: ").strip()
    if changes:
        response = review_travel_plan("test_user", "change", changes)
    else:
        response = review_travel_plan("test_user", "approve")

print("\nFINAL RESPONSE:\n")
print(response["answer"])
print(f"\n(LLM calls used: {response['llm_calls']})")