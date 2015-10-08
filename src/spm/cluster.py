from pylab import *
from sklearn import datasets
from sklearn.neighbors import NearestNeighbors
from collections import deque

def sum_up_links(sum_of_links, spins):
    llinks = arange(len(spins))
    for si in range(n_spins):
        s_idx = spins == si
        link_idx = llinks[s_idx]
        for pos in range(len(link_idx) - 1):
            i = link_idx[pos]
            for j in link_idx[pos + 1:]:
                sum_of_links[i, j] += 1

def hamilton(dij, n_neighbors, scaler):
    """
    :param dij: the distance between x_i and x_j
    :param k: number of nearest neighbors
    :param alpha: scaling factor in our case medium distance of nearest neighbors

    :result : probability value 
    """
    return 1. / n_neighbors * exp(-dij**2 / (scaler)**2)


def wolff_ising_step(L, C, handle_node, links, link_probs, spins):
    # get the index of the last entry
    idx = L.pop()
    handle_node[idx] = False
    C.append(idx)
    link_full = links[idx]
    # only hanlde the not visited 
    handle_idx = handle_node[link_full]
    link = link_full[handle_idx]
    probs_n_visit = link_probs[idx][handle_idx]

    # spin extraction only choose the nodes
    # where the spin is the same as in the head node
    si = spins[idx]
    ss_idx = spins[link] == si


    # get the link probabilities
    probs = probs_n_visit[ss_idx]

    is_linked = rand(len(probs)) < probs

    linked_idx = link[ss_idx][is_linked]
    return linked_idx


def wolff_ising(links, link_probs, spins, iterations=None):
    """
    With the Wolff ising algorithm spins a random set of nodes
    are used and go through the chain. There links where build
    and spins where flipped such as the total system should converge
    to a final spin set

    :param links: is a 2D array where the first index is
                  connected to the node and the second index
                  corresponds to the nearest neighbors
    :param link_prob: is the computed probability for the corresponding neighbor
                      from the links
    :param spins: is the current spins set for the nodes !!!! this parameter will
                  be changed by this method
    :param iterations: if no iterations is set then the number of
                       nodes are used
    """
    handle_node = array([True for _ in range(len(spins))])
    L = deque()
    C = deque()

    if iterations is None:
        iterations = len(spins)

    for i in range(iterations):
        # reset the nodes and the connectionlist
        handle_node[:] = True
        C.clear()

        # set the first node randomly
        idx = randint(len(links))
        L.append(idx)
        while(len(L) > 0):
            linked_idx = wolff_ising_step(L, C, handle_node, links, link_probs, spins)
            # if the idx is not visited
            for l in linked_idx:
                C.append(l)
            # flip spin
        # new spin
        new_spin = randint(n_spins)
        for c in C:
            spins[c] = new_spin


def swendson_wang(links, link_probs, spins):
    pass
## static parameter later for function 
n_spins = 20
n_samples = 1024
n_neighbors = 10
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.1)
X, y = noisy_moons

## nearest neighbors
neighbors = NearestNeighbors(n_neighbors=n_neighbors)
kn = neighbors.fit(X)
dists, links = kn.kneighbors()

graph = kn.kneighbors_graph()
spins = array([randint(n_spins) for _ in range(len(X))])


## compute parameters
med_distance = mean(dists)
N = 1
link_probs = hamilton(dists, N, med_distance)


## fill the probabilities
for i, link in enumerate(links):
    for pos_link, j in enumerate(link):
        id1 = graph[i, j]
        #assert(id1 == 1.)
        graph[i, j] = link_probs[i, pos_link]

## remove doubles
gx = graph * 0
count = 0
red = 0
for i, link in enumerate(links):
    for pos_link, j in enumerate(link):
        count += 1
        v1 = graph[i, j]
        v2 = graph[j, i]
        if v1 > 0 and v2 > 0:
            red += 1
        v = max(v1, v2)
        if i > j:
            gx[j, i] = v
        else:
            gx[i, j] = v

## wolff
from functools import partial
from multiprocessing import Pool


sum_of_links = zeros((n_samples, n_samples))
#sum_of_links = gx * 0


T = linspace(0, 3, 10)
betas = 1. / T



def run_magnetisation(beta, links, link_probs, spins, NUM = 1000, inner_iter=100):
    NUM = 1000

    m_sum = 0
    m_sqr = 0
    lb = exp(-(1-link_probs) * beta)
    print(lb)
    for j in range(NUM):
        #if j % (NUM // 10) == 0:
        #    print(j)
        wolff_ising(links, lb, spins, 100)
        #sum_up_links(sum_of_links, spins)
        Ni = [sum(spins == si) for si in range(n_spins)]
        Nm = max(Ni)
        m = ((float(Nm) / len(spins)) * n_spins - 1) / float(n_spins - 1)
        m_sum += m
        m_sqr += m**2
        #clf()
        #scatter(X[:, 0], X[:, 1], c=spins)
        #draw()

    res = (m_sqr - (m_sum**2) / NUM) / (NUM -1)
    print beta, res
    return beta, res

fun = partial(run_magnetisation,  links=links, link_probs=link_probs, spins=spins, NUM=1000)
p = Pool(2)
beta_chi = p.map(fun, betas)
beta, chi = array(beta_chi).T

plot(beta, chi * beta)


###############################################################################
## swendson wang
# simultanious link all nodes per iteration
###############################################################################


###############################################################################
## show links
###############################################################################
s = sum_of_links / NUM
NX, NY = where(s > 0.21)
llidx = arange(len(NX))

for i in range(1000):
    id_sel = randint(len(llidx))
    ids = llidx[id_sel]

    i = NX[ids]
    j = NY[ids]
    plot(X[(i, j), 0], X[(i, j), 1])
show()
