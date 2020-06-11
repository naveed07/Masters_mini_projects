
#All 28 dominoes of a traditional double-six set can be arranged in
#a cycle so that the spots of adjacent dominoes match each other. How many such cycles are possible?


temp_dominoes = [ [0,0], [0,1], [1,1], [0,2], [1,2], [2,2], [0,3], [1,3], [2,3],[3,3],
             [0,4], [1,4], [2,4], [3,4], [4,4], [0,5], [1,5], [2,5], [3,5],
             [4,5], [5,5], [0,6], [1,6], [2,6], [3,6], [4,6], [5,6], [6,6] ]


dom = temp_dominoes.copy()
exchange= 0
cnt =0
i =1
lc = 1
fe = 1
circle1 = []
cir_cnt = 0
#pre_ele = dom[0]
#cur_ele = dom[1]
while lc > 0 and exchange < 27:
    lc = len(dom)

    if lc == 1:
        if dom[0][1] == 0:
            circle1.append(dom[0])
            cir_cnt += 1
            print(cir_cnt)
            print(circle1)
            circle1 = []
            tep = temp_dominoes[exchange + 1]
            temp_dominoes[exchange + 1] = temp_dominoes[27 - exchange]
            temp_dominoes[27 - exchange] = tep
            dom = temp_dominoes.copy()
            exchange += 1
            if exchange == 27:
                print('there are ', cir_cnt, 'possible circles')

           # print(exchange)
        elif dom[0][0] == 0:
            tem = dom[0][0]
            dom[0][0] = dom[0][1]
            dom[0][1] = tem
            circle1.append(dom[0])
            cir_cnt += 1
            print(cir_cnt)
           # print(circle1)
            circle1= []
            tep = temp_dominoes[exchange + 1]
            temp_dominoes[exchange + 1] = temp_dominoes[27 - exchange]
            temp_dominoes[27 - exchange] = tep
            dom = temp_dominoes.copy()
            exchange += 1
            if exchange == 27:
                print('there are ', cir_cnt, 'possible circles')
           # print(exchange)
        else:
            print('no circle')
            tep = temp_dominoes[exchange + 1]
            temp_dominoes[exchange + 1] = temp_dominoes[27 - exchange]
            temp_dominoes[27 - exchange] = tep
            dom = temp_dominoes.copy()
            exchange += 1
    else:
        if dom[0][1] == dom[i][0] and fe == 1:
            if i != 1:
                dom[0] = dom[i]
                circle1.append(dom[0])
                del dom[i]
                i = 1
            else:
                del dom[0]
                circle1.append(dom[0])
            cnt += 1

        elif dom[0][1] == dom[i][1] and fe == 0:
            tem = dom[i][0]
            dom[i][0] = dom[i][1]
            dom[i][1] = tem
            if i != 1:
                dom[0] = dom[i]
                circle1.append(dom[0])
                del dom[i]
                i = 1
            else:
                circle1.append(dom[0])
                del dom[0]
            cnt += 1
            fe = 1
        elif i == len(dom)-1 and fe == 1:
            fe = 0
            i = 1
        elif i==len(dom)- 1 and fe == 0:
            tep = temp_dominoes[exchange + 1]
            temp_dominoes[exchange + 1] = temp_dominoes[27 - exchange]
            temp_dominoes[27 - exchange] = tep
            dom = temp_dominoes.copy()
            #print(circle1)
            circle1 = []
            exchange += 1
            if exchange == 27:
                print('there are ', cir_cnt, 'possible circles')
           # print(exchange)
           # print('no cycle')
        else:
            i += 1




