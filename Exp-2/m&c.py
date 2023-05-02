from colletions import deque
def is_safe(state):
     m, c = state
     return m >= 0 and c >= 0 and (m == 0 or m >= c)
def bfs(start, goal, q):
    q.append(start)
    while q:
        state = q.popleft()
        if state == goal:
            return state m, c = state
    for (mm, cc) in [(1, 0), (2, 0), (0, 1), (0, 2), (1, 1)]:
        next_state = (m - mm, c - cc)
        if is_safe(next_state):
            q.append(next_state)
            return None
start = (3, 3)
goal = (0, 0)
q = deque()
result = bfs(start, goal, q)
if result:
    print("Found a solution:", result)
else:
    print("No solution found")

