

# Give all the pairs of functions o and ☐ (from 16 Booleans), for which (xoy)☐z=(x☐z)o(y☐z).

# there are 16 boolean function and 3 variable  which gives 16 * 16 = 256 cases for each 8 cases
# total 16*16 = 256 and 256 * 8 = 2048 cases

def Contradiction(a,b):
    return 0

def Tautology(a,b):
    return 1

def Propositionp(a,b):
    if a == 0:
        return  0
    elif a == 1:
        return 1

def Negationp(a,b):
    if a == 0:
        return  1
    elif a == 1:
        return 0

def Propositionq(a,b):
    if b == 0:
        return  0
    elif b == 1:
        return 1

def Negationq(a,b):
    if a == 0:
        return  1
    elif a == 1:
        return 0

def Conjunction(a,b):
    if a == b == 1:
        return  1
    else:
        return 0

def Alternative_denial(a,b):
    if a == b == 1:
        return  0
    else:
        return 1

def Disjunction(a,b):
    if a == b == 0:
        return  0
    else:
        return 1

def Joint_denial(a,b):
    if a == b == 0:
        return  1
    else:
        return 0

def Material_nonimplication(a,b):
    if a == 1 and b == 0:
        return 1
    else:
        return 0

def Material_implication(a,b):
    if a == 1 and b == 0:
        return 0
    else:
        return 1

def Converse_nonimplication(a,b):
    if a == 0 and b == 1:
        return 1
    else:
        return 0

def Converse_implication(a,b):
    if a == 0 and b == 1:
        return 0
    else:
        return 1

def Exclusive_disjunction(a,b):
    if a == b:
        return  0
    else:
        return 1
def Biconditional(a,b):
    if a == b:
        return  1
    else:
        return 0
count = 0
actual_count = 0
functions = [Contradiction, Tautology, Propositionp, Negationp, Propositionq, Negationq,Conjunction,
             Alternative_denial,Disjunction,Joint_denial,Material_nonimplication,Material_implication,Converse_nonimplication,
             Converse_implication,Exclusive_disjunction,Biconditional]
print_func = ['Contradiction', 'Tautology', 'Propositionp', 'Negationp', 'Propositionq','Negationq','Conjunction'
              ,'Alternative_denial','Disjunction','Joint_denial','Material_nonimplication','Material_implication','Converse_nonimplication'
              ,'Converse_implication','Exclusive_disjunction','Biconditional']
x = [1,1,1,1,0,0,0,0]
y = [1,1,0,0,1,1,0,0]
z = [1,0,1,0,1,0,1,0]
combination = []
t = [0 , 0]
for i in range(8): # u can change this parameter to 8 to find all 2048 cases
    j= x[i]
    k= y[i]
    l= z[i]

    for m in range(16):
        for n in range(16):
            if m != n:
                function = functions[m]
                left_temp = function(j,k)
                function = functions[n]
                left_final = function(left_temp, l)

                function = functions[m]
                right_t1 = function(j, l)
                function = functions[m]
                right_t2 = function(k, l)
                function = functions[n]
                right_final = function(right_t1, right_t2)
                actual_count +=1

                if left_final == right_final:
                    #print( '( (', j ,print_func[m], k ,')', print_func[n], l,')  ==  (  ( ', j, print_func[m], l,')',print_func[n],'(',k, print_func[m], l, ') )' )
                    t[0] = print_func[m]
                    t[1] = print_func[n]

                    if t not in combination:
                        combination.append(t)
                        t = [0,0]
                        count += 1
                        print(print_func[m],print_func[n])
print( count, ' combinations satisfies the distributive property ' )