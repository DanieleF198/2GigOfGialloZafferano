import clingo

maxp = 15

def better(m1_cost,m2_cost):
    for i in range(len(m1_cost)):
        if (m1_cost[i] < m2_cost[i]):
            return 1
        if (m2_cost[i] < m1_cost[i]):
            return 2
    return 0

def create_preamble(max_priority_level):
    out = ""
    for i in range(max_priority_level):
        out = out + ":~ #true. [0@{}]\n".format(i+1)
    return out

def compare(i1, i2, weak_constraints):
    preamble = create_preamble(maxp)

    ctl1 = clingo.Control()
    ctl1.add("base", [], preamble)
    ctl1.add("base", [], weak_constraints)
    ctl1.add("base", [], i1)
    ctl1.ground([("base", [])])

    with ctl1.solve(yield_=True) as handle:
        for m in handle:
            m1_cost = m.cost

    ctl2 = clingo.Control()
    ctl2.add("base", [], preamble)
    ctl2.add("base", [], weak_constraints)
    ctl2.add("base", [], i2)
    ctl2.ground([("base", [])])

    with ctl2.solve(yield_=True) as handle:
        for m in handle:
            m2_cost = m.cost

    return better(m1_cost, m2_cost)
