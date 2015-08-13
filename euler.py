import sys
import time
from collections import defaultdict, Counter
from math import factorial, log10
from itertools import permutations, combinations, cycle, compress
from operator import mul
from fractions import Fraction, gcd


def is_palindrome(n):
    return str(n) == str(n)[::-1]


def is_prime(n):
    if n < 2 or n % 2 == 0:
        return n == 2
    for i in xrange(3, int(n**0.5) + 1, 2):
        if n % i == 0:
            return False
    return True


def divisors(n):
    for i in xrange(1, int(n**0.5) + 1, 1 if n % 2 == 0 else 2):
        if n % i == 0:
            yield i
            if i != n / i:
                yield n / i


def abundant_numbers(n):
    for i in xrange(1, n + 1):
        if sum(divisors(i))/2 > i:
            yield i


def solve_quadratic(a, b, c):
    return (-b + (b**2 - 4 * a * c)**.5) / (2 * a)


def is_triangle(c):
    return solve_quadratic(1, 1, 2 * -c).is_integer()


def is_square(c):
    return solve_quadratic(1, 0, -c).is_integer()


def is_pentagonal(c):
    return solve_quadratic(3, -1, 2 * -c).is_integer()


def is_hexagonal(c):
    return solve_quadratic(2, -1, -c).is_integer()


def is_heptagonal(c):
    return solve_quadratic(5, -3, 2 * -c).is_integer()


def is_octagonal(c):
    return solve_quadratic(3, -2, -c).is_integer()


def p1():
    return sum(i for i in xrange(1000) if i % 3 == 0 or i % 5 == 0)


def p2():
    total, x, y = 0, 0, 1
    while y <= 4000000:
        total += y if y % 2 == 0 else 0
        x, y = y, x + y
    return total


def p3():
    i, n = 3, 600851475143
    while n != i:
        while n % i == 0:
            n /= i
        i += 2
    return n


def p7():
    n, z = 0, 0
    while n < 10001:
        z += 1
        n += 1 if is_prime(z) else 0
    return z


def p20():
    return sum(map(int, str(reduce(lambda x, y: x * y, xrange(2, 101)))))


def p22():
    with open('names.txt', 'r') as f:
        name = sorted(f.readline().strip('"').split('","'))
    return sum((i+1) * sum(ord(c) - ord('A') + 1 for c in name) for i, name in enumerate(name))


def p29():
    return len(set(x**y for x in xrange(2, 101) for y in xrange(2, 101)))


def p30():
    return sum(x for x in xrange(1000, 5*(9**5)) if x == sum(int(y)**5 for y in str(x)))


def p31():
    l = [1, 2, 5, 10, 20, 50, 100, 200]

    def f(left, l):
        if sum(left) == 200:
            return 1
        return sum(f(left + [v], l[i:]) for i, v in enumerate(l) if sum(left) + v <= 200)
    return sum(f([v], l[i:]) for i, v in enumerate(l))


def p32():
    t = set()
    for i in permutations(range(1, 10) + ['|']*2, len(range(1, 12))):
        m1, m2, p = ''.join(map(str, i)).split('|')
        if m1 and m2 and p and int(m1) * int(m2) == int(p):
            t.add(int(p))
    return sum(t)


def p33():
    arr = []
    for i in xrange(10, 100):
        for j in xrange(10, 100):
            if i < j:
                i1, i2 = str(i)
                j1, j2 = str(j)
                if i2 == j1 and j2 != '0' and float(i)/float(j) == float(i1)/float(j2):
                    arr.append((i, j))
    return Fraction(*map(lambda x: reduce(mul, x, 1), zip(*arr)))


def p34():
    return sum(i for i in xrange(3, 1000000) if i == sum(map(factorial, map(int, str(i)))))


def p35():
    s, t = set('2'), set()
    for i in xrange(3, 1000000, 2):
        if str(i) not in t and str(i) not in s:
            r = [str(i)[j:] + str(i)[:j] for j in range(len(str(i)))]
            s.update(r) if all(map(is_prime, map(int, r))) else t.update(r)
    return len(s)


def p36():
    return sum(i for i in xrange(1000000) if is_palindrome(i) and is_palindrome(bin(i)[2:]))


def p37():

    def truncates(s):
        for i in range(len(s)):
            yield s[i:]
            if i > 0:
                yield s[:-i]

    s, t = set(), set()
    for i in xrange(11, 1000000, 2):
        if str(i) not in t and str(i) not in s:
            r = list(truncates(str(i)))
            s.add(i) if all(map(is_prime, map(int, r))) else t.add(i)
    return sum(s)


def p38():
    c = 0
    for i in xrange(10000):
        j = 1
        s = ''
        while True:
            if len(str(i * j)) != len(set(str(i * j))) or set(s) & set(str(i * j)):
                break
            s += str(i * j)
            if set(s) == set(map(str, xrange(1, 10))) and int(s) > c:
                c = int(s)
            j += 1
    return c


def p39():
    d = defaultdict(set)
    for i in xrange(1, 1001//2):
        for j in xrange(i, 1001//2):
            c = (i**2 + j**2)**.5
            if c.is_integer():
                p = i + j + c
                if p <= 1000:
                    d[p].add(tuple(sorted((i, j, c))))
    z = dict((k, len(v)) for k, v in d.iteritems())
    return max(z, key=z.get)


def p40():
    i, j, p = 0, 0, 1
    while j < 1000000:
        i += 1
        for n in xrange(1, len(str(i)) + 1):
            if log10(j + n).is_integer():
                p *= int(str(i)[n - 1])
        j += len(str(i))
    return p


def p41():
    q = []
    for i in range(2, 10):
        p = map(int, map(lambda x: ''.join(map(str, x)), permutations(range(1, i + 1), i)))
        q += [z[0] for z in zip(p, map(is_prime, p)) if z[1]]
    return max(q)


def p42():
    with open('words.txt') as f:
        words = map(lambda x: x.strip('"'), f.read().split(','))
    n = [sum(ord(l) - 64 for l in word) for word in words]
    p, t = 2, [1]
    while t[-1] < max(n):
        t.append(int(p * .5 * (p + 1)))
        p += 1
    return sum(1 for i in n if i in t)


def p43():
    p = [2, 3, 5, 7, 11, 13, 17]
    t = 0
    for i in map(lambda x: ''.join(map(str, x)), permutations(range(10), 10)):
        if all(int(i[j + 1:j + 4]) % p[j] == 0 for j in range(7)):
            t += int(i)
    return t


def p44():
    for p1, p2 in combinations((n * (3 * n - 1)/2 for n in range(1, 10000)), 2):
        if is_pentagonal(p2 - p1) and is_pentagonal(p1 + p2):
            return p2 - p1


def p45():
    def triangles(i=0):
        while True:
            i += 1
            yield (i * (i + 1)) / 2

    for t in triangles(285):
        if is_pentagonal(t) and is_hexagonal(t):
            return t


def p46():
    i, g = 33, True
    while g:
        i += 2
        if is_prime(i):
            continue
        g = False
        for j in range(1, int((i/2)**0.5) + 1):
            if is_prime(i - 2 * j**2):
                g = True
                continue
    return i


def p47():
    n = 209
    while True:
        n += 1
        if all(sum(1 for i in divisors(n + j) if is_prime(i)) == 4 for j in range(4)):
            return n


def p48():
    return str(sum(i**i for i in range(1, 1001)))[-10:]


def p49():
    d = {}
    for p in range(1000, 10000):
        x = set()
        for q in permutations(tuple(str(p)), 4):
            if q[0] == '0' or q in d:
                continue
            d[q] = True
            q2 = int(''.join(map(str, q)))
            if is_prime(q2):
                x.add(q2)
        if len(x) > 2:
            dd = defaultdict(set)
            for r, s in combinations(x, 2):
                dd[s - r].update((r, s))
                if len(dd[s - r]) == 3:
                    return dd[s - r]


def p50():
    def primes():
        i = 2
        while True:
            if is_prime(i):
                yield i
            i += 1

    a = []
    y, z = 0, 0
    for i in primes():
        for b in xrange(len(a)):
            a[b] += i
            if is_prime(a[b]):
                x = len(a) - b + 1
                if x > y:
                    if a[b] > 1000000:
                        return z
                    y = x
                    z = a[b]
        a.append(i)


def p439(n, o=0):
    f = {}
    s = 0
    for i in xrange(o + 1, n + 1):
        for j in xrange(1, i):
            x = i * j
            if x in f:
                s += f[x]
            else:
                y = sum(divisors(i*j))
                f[x] = y
                s += y
            s = s % 10**9
    s *= 2
    s += sum(sum(divisors(i**2)) % 10**9 for i in range(o + 1, n + 1))
    return s


def primes(limit):
    i = 2
    while i < limit:
        if is_prime(i):
            yield i
        i += 1

p = list(primes(10))


def p439a(x, y, z):
    if y == 3:
        return x
    for i in p:
        x += p439a([[i, y]], y + 1, z + 1)
    return x


def p51():
    for i in xrange(1, 10000, 2):
        for t in xrange(1, len(str(i)) + 1):
            z = set()
            for j in permutations(str(i) + '*' * t):
                if j[0] == '0' or j[-1] == '*' or j in z:
                    continue
                z.add(j)
                b, c = 0, []
                for k in xrange(10):
                    if k == 0 and j[0] == '*':
                        b += 1
                        continue
                    candidate = ''.join(j).replace('*', str(k))
                    if not is_prime(int(candidate)):
                        b += 1
                        if b > 2:
                            continue
                    c.append(candidate)
                if b == 2:
                    return c[0]


def p52():
    for i in xrange(1, 1000000):
        x = Counter(str(i))
        for j in xrange(2, 7):
            if x != Counter(str(i * j)):
                break
        if j == 6:
            return i


def p53():
    def c(n, r):
        return factorial(n)/(factorial(r) * factorial(n - r))

    return sum(1 for i in range(23, 101) for j in range(i) if c(i, j) > 1000000)


def p54():
    def is_consecutive(values):
        return all(values[i] - 1 == values[i + 1] for i in xrange(len(values) - 1))

    def is_same(suits):
        return all(suits[0] == item for item in suits)

    def get_rank(hand):
        values, suits = zip(*hand)
        translations = dict(zip(('T', 'J', 'Q', 'K', 'A'), range(10, 15)))
        values = sorted(
            (translations[value] if value in translations else int(value) for value in values),
            reverse=True)
        counter = Counter(values)

        if is_consecutive(values) and is_same(suits):  # Straight flush
            return [8, values[0]]
        if sorted(counter.values()) == [1, 4]:  # Four of a kind
            return [7, counter.most_common(1)[0][0]]
        if sorted(counter.values()) == [2, 3]:  # Full house
            return [6, counter.most_common(1)[0][0]]
        if is_same(suits):  # Flush
            return [5, values[0]]
        if is_consecutive(values):  # Straight
            return [4, values[0]]
        if sorted(counter.values()) == [1, 1, 3]:  # Three of a kind
            return [3, counter.most_common(1)[0][0]]
        if sorted(counter.values()) == [1, 2, 2]:  # Two pairs
            return [2] + sorted((i[0] for i in counter.most_common(2)), reverse=True) + [counter.most_common(3)[2][0]]
        if sorted(counter.values()) == [1, 1, 1, 2]:  # One pair
            pair = counter.most_common(1)[0][0]
            cards = [v for v in values if v != pair]
            return [1, pair] + cards
        return [0] + values  # High card

    wins = 0
    with open('poker.txt', 'r') as f:
        for line in f:
            hands = line.strip().split(' ')
            wins += 1 if get_rank(hands[:5]) > get_rank(hands[5:]) else 0
    return wins


def p55():
    l = 0
    for i in range(1, 10000):
        t = i
        for j in range(51):
            s = list(str(t))
            s.reverse()
            t += int(''.join(s))
            if is_palindrome(t):
                break
        if j == 50:
            l += 1
    return l


def p56():
    t = 0
    for a in xrange(2, 100):
        for b in xrange(2, 100):
            s = sum(map(int, list(str(a**b))))
            t = s if s > t else t
    return t


def p57():
    t = 0
    for i in range(1, 1001):
        e = Fraction(0)
        for j in range(i):
            e = Fraction(1)/(Fraction(2) + e)
        e = Fraction(1) + e
        t += 1 if len(str(e.numerator)) > len(str(e.denominator)) else 0
    return t


def p58():
    x, y, primes = 49, 8, 8
    while primes / float(y * 2 - 1) > 0.10:
        for i in xrange(4):
            x += y
            primes += 1 if is_prime(x) else 0
        y += 2
    return y - 1


def p59():
    with open('cipher1.txt', 'r') as f:
        text = f.readline().strip().split(',')
    d = [[], [], []]
    for i, v in enumerate(text):
        d[i % 3].append(v)
    key = [int(Counter(a).most_common(1)[0][0]) ^ ord(' ') for a in d]
    key = cycle(key)
    plain_text = [int(i) ^ key.next() for i in text]
    return sum(plain_text)


def p60():
    a, i = {3: []}, 5
    while True:
        i += 2
        if not is_prime(i):
            continue
        for x, y in a.iteritems():
            if is_prime(int('%d%d' % (x, i,))) and is_prime(int('%d%d' % (i, x,))):
                for j in y:
                    if all(is_prime(int('%d%d' % (i, k,))) and is_prime(int('%d%d' % (k, i,))) for k in j):
                        j.append(i)
                        if len(j) == 4:
                            return sum(j) + x
                y.append([i])
        a[i] = []


def p61():
    fs = [is_triangle, is_square, is_pentagonal, is_hexagonal, is_heptagonal, is_octagonal]

    d = defaultdict(list)
    e = {}
    octagonals = []
    for i in xrange(1000, 10000):
        figurates = [f(i) for f in fs]
        if figurates[-1]:
            octagonals.append(i)
        if any(figurates[:-1]):
            d[i//100].append(i)
            e[i] = list(compress(xrange(3, 8), figurates[:-1]))

    def cyc(left):
        if len(left) == 6:
            if left[-1] % 100 == left[0]//100:
                return [left]
            return []
        else:
            r = []
            for v in d[left[-1] % 100]:
                if v in left:
                    continue
                x = cyc(left + [v])
                r.extend(x)
            return r

    l = []
    for j in octagonals:
        cycles = cyc([j])
        for a in cycles:
            z = [m for k in a[1:] for m in e[k]]
            if all(i in z for i in xrange(3, 8)):
                return sum(a)


def p62():
    cubes = defaultdict(list)
    key, i = None, 346
    while len(cubes[key]) < 5:
        key = ''.join(sorted(str(i**3)))
        cubes[key].append(i**3)
        i += 1
    return min(cubes[key])


def p63():
    c = 0
    for i in xrange(1, 10):
        j = 1
        while len(str(i**j)) >= j:
            c, j = c + 1, j + 1
    return c


def p64():

    def odd_period(n):
        a0 = int(n**0.5)
        numerator, denominator = 1, a0
        is_odd = False
        while True:
            a = int(numerator/(n**0.5 - denominator))
            numerator = (n - denominator**2)/numerator
            denominator = -(denominator - a * numerator)
            is_odd = not is_odd
            if numerator == 1 and denominator == a0:
                return is_odd
    return sum(1 for i in xrange(2, 10001) if not (i**0.5).is_integer() and odd_period(i))


def p65():
    z = reversed([2 * (i//3 + 1) if i % 3 == 1 else 1 for i in xrange(99)])
    e = Fraction(0)
    for i in z:
        e = Fraction(1)/(Fraction(i) + e)
    e = Fraction(2) + e
    return sum(map(int, str(e.numerator)))


def p66():

    def period(n):
        a0 = int(n**0.5)
        yield a0
        numerator, denominator = 1, a0
        while True:
            a = int(numerator/(n**0.5 - denominator))
            yield a
            numerator = (n - denominator**2)/numerator
            denominator = -(denominator - a * numerator)
    largest_x, _d = 0, 0
    for d in xrange(2, 1001):
        if (d**0.5).is_integer():
            continue
        p = period(d)
        n2, n1 = 1, p.next()
        d2, d1 = 0, 1
        while n1**2 - d * d1**2 != 1:
            n = p.next()
            n1, n2 = n1 * n + n2, n1
            d1, d2 = d1 * n + d2, d1
        if n1 > largest_x:
            largest_x, _d = n1, d
    return _d


def p68():
    def is_same(l):
        return all(sum(l[0]) == sum(item) for item in l)

    outer_possibilities = [1, 2, 3, 4, 5, 7, 8, 9]
    solution = ''
    for outer in combinations(outer_possibilities, 3):
        inner = [i for i in outer_possibilities if i not in set(outer)]
        outer += (10,)
        for o in permutations(outer):
            if min(o) < 6:
                continue
            o = (6,) + o
            for i in permutations(inner):
                i += i
                solution_set = [(b, i[a], i[a + 1],) for a, b in enumerate(o)]
                if is_same(solution_set):
                    s = ''.join([str(j) for i in solution_set for j in i])
                    solution = s if s > solution else solution
    return solution


def p69():
    l = [i - 1 for i in xrange(1000001)]
    for i in xrange(2, 1000001):
        for j in xrange(2, (1000000 // i) + 1):
            l[i * j] -= l[i]
    return max(((n+2.0)/j, n+2,) for n, j in enumerate(l[2:]))[1]


def p70():
    l = [i - 1 for i in xrange(10000001)]
    for i in xrange(2, 10000001):
        for j in xrange(2, (10000000 // i) + 1):
            l[i * j] -= l[i]
    return min(((n+2.0)/j, n+2,) for n, j in enumerate(l[2:]) if sorted(str(n+2)) == sorted(str(j)))[1]


def p71():
    largest_multiple = 1000000 - 1000000 % 7
    numerator = largest_multiple / 7 * 3
    while gcd(numerator, largest_multiple) != 1:
        numerator -= 1
    return numerator


def p72():
    l = [i - 1 for i in xrange(1000001)]
    for i in xrange(2, 1000001):
        for j in xrange(2, (1000000 // i) + 1):
            l[i * j] -= l[i]
    return sum(l) + 1


def p73():
    return sum(1 for i in xrange(5, 12001) for j in xrange(i/3+1+(i/3) % (2-i % 2), i/2+1, 2-i % 2) if gcd(i, j) == 1)


def p74():
    z, t, = {}, 0
    for i in xrange(2, 1000000):
        i = ''.join(sorted(list(str(i))))
        c, s = [], set()
        next_term = i
        while next_term not in s:
            if next_term in z:
                c += z[next_term]
                break
            s.add(next_term)
            c.append(next_term)
            next_term = sum(factorial(int(i)) for i in str(next_term))
            next_term = ''.join(sorted(list(str(next_term))))
        z[i] = c
        if len(c) == 60:
            t += 1
    return t


def p75():
    def f(m, n):  # Euclid's formula for pythagorean triples
        return m**2 - n**2, 2 * m * n, m**2 + n**2

    d = defaultdict(int)
    m, n = 2, 2
    while n > 1 or m < 4:
        for n in xrange(1, m):
            if (m-n) % 2 != 1 or gcd(m, n) != 1:
                continue
            triple = f(m, n)
            l = sum(triple)
            if l > 1500000:
                break
            k = 1
            while l <= 1500000:
                d[l] += 1
                k += 1
                l = sum(map(lambda i: k*i, triple))
        m += 1
    return sum(1 for v in d.values() if v == 1)


def p76():

    def generalized_pentagonal(n=0):
        while True:
            k = -n/2
            if n % 2:
                k = (n + 1)/2
            yield (k, (k*(3*k - 1))/2,)
            n += 1

    partitions = [1]

    def partition(n):
        if n < len(partitions):
            return partitions[n]
        gpg = generalized_pentagonal(2)
        total, k, gp = 0, 1, 1
        while gp <= n:
            total += int((-1)**(k-1)) * partition(n-gp)
            k, gp = gpg.next()
        partitions.insert(n, total)
        return total
    return partition(100) - 1


def p77():
    def primes():
        yield 2
        n = 3
        while True:
            if is_prime(n):
                yield n
            n += 2

    def q(n):
        partitions = defaultdict(int)
        partitions[0] = 1
        for i in primes():
            for j in xrange(n-i+1):
                partitions[i+j] += partitions[j]
            if i > n:
                return partitions[n]
    n = 10
    while q(n) < 5000:
        n += 1
    return n


def p78():
    def generalized_pentagonal(n=0):
        while True:
            k = -n/2
            if n % 2:
                k = (n + 1)/2
            yield (k, (k*(3*k - 1))/2,)
            n += 1

    partitions = [1]

    def partition(n):
        if n < len(partitions):
            return partitions[n]
        gpg = generalized_pentagonal(2)
        total, k, gp = 0, 1, 1
        while gp <= n:
            total += int((-1)**(k-1)) * partition(n-gp)
            k, gp = gpg.next()
        partitions.insert(n, total)
        return total
    n = 10
    while partition(n) % 1000000:
        n += 1
    return n


def p79():
    appearances = defaultdict(list)
    with open('keylog.txt', 'r') as f:
        for line in f:
            for i, n in enumerate(line.strip()):
                appearances[n].append(i)

    average_positions = {}
    for k, v in appearances.items():
        average_positions[k] = float(sum(v))/float(len(v))

    a = [k for k, v in sorted(average_positions.items(), key=lambda a: a[1])]
    return ''.join(str(x) for x in a)


def p80():
    def f(c):
        p = '0'
        for i in xrange(100):
            x = 0
            while (20*int(p) + x) * x <= c:
                x += 1
            x -= 1
            c -= (20*int(p) + x) * x
            if c == 0:
                return 0
            c *= 100
            p += str(x)
        return sum(int(i) for i in p)
    return sum(f(i) for i in xrange(100))


def p97():
    a = 1
    for i in range(7830457):
        a = 2 * a % 10000000000
    return (28433 * a + 1) % 10000000000


def p206():
    value = int(1929394959697989990**0.5)
    value -= 30 + value % 100
    c = cycle([40, 60])
    while not str(value**2)[::2] == '1234567890':
        value -= c.next()
    return value


def p448():
    D = 999999017

    def lcm(x, y):
        return (x / gcd(x, y)) * y

    def a(n):
        return sum(lcm(n, i) for i in xrange(1, n + 1))/n

    def b(n):
        return sum(lcm(n, i) % D for i in xrange(1, n + 1))/n % D
    return sum(a(i) % D for i in xrange(1, 99999999019+1))

if __name__ == '__main__':
    start_time = time.time()
    print locals()[sys.argv[1]]()
    print "Ran in %f seconds" % (time.time() - start_time)
