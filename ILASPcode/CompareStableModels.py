import clingo

maxp = 5


def relation_satisfied_case_no_zero(m1_cost,m2_cost, sign):
    if sign == '<':     # m1 it's not preferred over m2
        for i in range(len(m1_cost)):
            if (m1_cost[i] < m2_cost[i]):
                return 1
            if (m2_cost[i] < m1_cost[i]):
                return 2
    else:   # m2 it's preferred over m1
        for i in range(len(m1_cost)):
            if (m1_cost[i] < m2_cost[i]):
                return 2
            if (m2_cost[i] < m1_cost[i]):
                return 1
    return 0


def relation_satisfied(m1_cost,m2_cost, sign):
    if sign == '<':     # m1 it's not preferred over m2
        for i in range(len(m1_cost)):
            if (m1_cost[i] < m2_cost[i]):
                return 1
            if (m2_cost[i] < m1_cost[i]):
                return 2
    elif sign == '>':   # m2 it's preferred over m1
        for i in range(len(m1_cost)):
            if (m1_cost[i] < m2_cost[i]):
                return 2
            if (m2_cost[i] < m1_cost[i]):
                return 1
    else:   # m1 and m2 are indifferent
        for i in range(len(m1_cost)):
            if not (m1_cost[i] == m2_cost[i]):
                return 2
        return 1
    return 2

def relation_satisfied_cm(m1_cost,m2_cost, sign):
    if sign == '<':     # m1 it's not preferred over m2
        for i in range(len(m1_cost)):
            if m1_cost[i] < m2_cost[i]:
                return 1
            if m2_cost[i] < m1_cost[i]:
                return 2
        return 3
    elif sign == '>':   # m2 it's preferred over m1
        for i in range(len(m1_cost)):
            if m1_cost[i] < m2_cost[i]:
                return 4
            if m2_cost[i] < m1_cost[i]:
                return 5
        return 6
    else:   # m1 and m2 are indifferent
        for i in range(len(m1_cost)):
            if not (m1_cost[i] == m2_cost[i]):
                if m1_cost[i] < m2_cost[i]:
                    return 7
                else:
                    return 8
        return 9

def relation_satisfied_cm_grid(m1_cost,m2_cost, sign,  treshold_value):
    m1_total_cost = sum(m1_cost)
    m2_total_cost = sum(m2_cost)
    inside_treshold = False
    if abs(m1_total_cost) > abs(m2_total_cost) and (abs(m1_total_cost) - abs(m2_total_cost) < treshold_value):
        inside_treshold = True
    elif abs(m1_total_cost) < abs(m2_total_cost) and (abs(m2_total_cost) - abs(m1_total_cost) < treshold_value):
        inside_treshold = True
    elif m1_total_cost == m2_total_cost:
        inside_treshold = True
    if sign == '<':     # m1 it's not preferred over m2
        if inside_treshold:
            return 3
        else:
            if m1_total_cost < m2_total_cost:
                return 1
            else:
                return 2
    elif sign == '>':   # m2 it's preferred over m1
        if inside_treshold:
            return 6
        else:
            if m1_total_cost < m2_total_cost:
                return 4
            else:
                return 5
    else:   # m1 and m2 are indifferent
        if inside_treshold:
            return 9
        else:
            if m1_total_cost < m2_total_cost:
                return 7
            else:
                return 8


def create_preamble(max_priority_level):
    out = ""
    for i in range(max_priority_level):
        out = out + ":~ #true. [0@{}]\n".format(i+1)
    return out


def compare(i1, i2, sign, weak_constraints):
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

    return relation_satisfied(m1_cost, m2_cost, sign)


def compare_cm_grid(i1, i2, sign, weak_constraints, treshold_value, factors_combination):
    preamble = create_preamble(maxp)

    list_weak_constraint = weak_constraints.split('\n')
    list_weak_constraint = list_weak_constraint[:-1]

    while True:
        if len(list_weak_constraint) == 5:
            break
        else:
            list_weak_constraint.append("void")

    # for wc in list_weak_constraint:
    #     print(wc)

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

    macro_ingredients_dictionary = {"cereali": 0.0,
                                    "latticini": 1.0,
                                    "uova": 4.0,
                                    "farinacei": 7.0,
                                    "frutta": 2.0,
                                    "erbe_spezie_e_condimenti": 11.0,
                                    "carne": 5.0,
                                    "funghi_e_tartufi": 3.0,
                                    "pasta": 5.0,
                                    "pesce": 9.0,
                                    "dolcificanti": 1.0,
                                    "verdure_e_ortaggi": 8.0}

    preparation_dictionary = {"bollitura": 5.0,
                              "rosolatura": 5.0,
                              "frittura": 5.0,
                              "marinatura": 3.0,
                              "mantecatura": 4.0,
                              "forno": 5.0,
                              "cottura_a_fiamma": 5.0,
                              "cottura_a_vapore": 5.0,
                              "stufato": 5.0}

    for i, penalty_score in enumerate(m1_cost):   # note that the vector codification made by clingo is [5, 4, 3, 2, 1] while yours is [1, 2, 3, 4, 5]
        if "category" in list_weak_constraint[len(list_weak_constraint)-i-1] and not ("V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]):
            m1_cost[i] = float(m1_cost[i]) * factors_combination[len(list_weak_constraint)-i-1]
            # print("category")
        elif "prepTime" in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
            m1_cost[i] = (m1_cost[i] / 280)  * factors_combination[len(list_weak_constraint)-i-1]
            # print("prepTime")
        elif "void" in list_weak_constraint[len(list_weak_constraint)-i-1]:
            m1_cost[i] = float(m1_cost[i])
        else:
            bypass = False
            for ingredient in macro_ingredients_dictionary.keys():
                if bypass:
                    break
                if ingredient in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
                    m1_cost[i] = (m1_cost[i] / macro_ingredients_dictionary[ingredient]) * factors_combination[len(list_weak_constraint)-i-1]
                    bypass = True
            for preparation in preparation_dictionary.keys():
                if bypass:
                    break
                if preparation in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
                    m1_cost[i] = (m1_cost[i] / preparation_dictionary[preparation]) * factors_combination[len(list_weak_constraint)-i-1]
                    bypass = True
            # print("others")

    for i, penalty_score in enumerate(m2_cost):   # note that the vector codification made by clingo is [5, 4, 3, 2, 1] while yours is [1, 2, 3, 4, 5]
        if "category" in list_weak_constraint[len(list_weak_constraint)-i-1] and not ("V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]):
            m2_cost[i] = float(m2_cost[i]) * factors_combination[len(list_weak_constraint)-i-1]
            # print("category")
        elif "prepTime" in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
            m2_cost[i] = (m2_cost[i] / 280) * factors_combination[len(list_weak_constraint)-i-1]
            # print("prepTime")
        elif "void" in list_weak_constraint[len(list_weak_constraint)-i-1]:
            m2_cost[i] = float(m2_cost[i])
        else:
            bypass = False
            for ingredient in macro_ingredients_dictionary.keys():
                if bypass:
                    break
                if ingredient in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
                    m2_cost[i] = (m2_cost[i] / macro_ingredients_dictionary[ingredient]) * factors_combination[len(list_weak_constraint)-i-1]
                    bypass = True
            for preparation in preparation_dictionary.keys():
                if bypass:
                    break
                if preparation in list_weak_constraint[len(list_weak_constraint)-i-1] and "V1@" in list_weak_constraint[len(list_weak_constraint)-i-1]:
                    m2_cost[i] = (m2_cost[i] / preparation_dictionary[preparation]) * factors_combination[len(list_weak_constraint)-i-1]
                    bypass = True
            # print("others")

    return relation_satisfied_cm_grid(m1_cost, m2_cost, sign, treshold_value)

def compare_cm(i1, i2, sign, weak_constraints):
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

    return relation_satisfied_cm(m1_cost, m2_cost, sign)

def compare_case_no_zero(i1, i2, sign, weak_constraints):
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

    return relation_satisfied_case_no_zero(m1_cost, m2_cost, sign)
