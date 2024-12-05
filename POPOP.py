import numpy as np
import os
import random
import copy

class Individual:
    def __init__(self, pos, values):
        # pos: location and equal to x * w + y
        self.pos = pos
        self.values = values
        assert self.pos[0].shape[0] == self.values[0].shape[0]

    def cal_fitness(self, params, prev_conf): # change this (addition variables: prev_conf)
        perturbed_img = self.generate_img(params)

        sub = perturbed_img.flatten() - params["image"].flatten()
        self.l2 = np.linalg.norm(sub, ord = 2)
        self.l0 = np.linalg.norm(sub, ord = 0)

        self.num_perturbed = np.sum([
            np.count_nonzero([np.sum(np.abs(v)) for v in vv])
            for vv in self.values
        ])

        self.is_adv, self.loss, self.score_list, self.adver_list, self.f_score_list = params["loss"](
            (perturbed_img * 255).astype(np.uint8), self.values, prev_conf
        )

    def cal_norm(self, params):
        perturbed_img = self.generate_img(params)

        sub = perturbed_img.flatten() - params["image"].flatten()
        self.l2 = np.linalg.norm(sub, ord = 2)
        self.l0 = np.linalg.norm(sub, ord = 0)

        self.num_perturbed = np.sum([
            np.count_nonzero([np.sum(np.abs(v)) for v in vv])
            for vv in self.values
        ])

    def generate_img(self, params):
        perturbed_img = params["image"].copy()
        for j, box in enumerate(params["boxes"]):
            w = abs(box[2] - box[0])
            h = abs(box[3] - box[1])

            for i in range(len(self.pos[j])):
                posH = min(box[1] + self.pos[j][i] // w, params["image"].shape[0] - 1)
                posW = min(box[0] + self.pos[j][i] % w, params["image"].shape[1] - 1)

                perturbed_img[int(posH), int(posW)] += self.values[j][i]

        perturbed_img = np.clip(perturbed_img, 0, 1)
        return perturbed_img

    def __lt__(self, other):
        # if self.is_adv == other.is_adv:
        #     return self.loss < other.loss
        # return self.is_adv
        if self.is_adv:
            if not other.is_adv:
                return True
            else:
                return self.l2 < other.l2
                # return self.l0 < other.l0
                # return self.num_perturbed < other.num_perturbed

        else:
            if other.is_adv:
                return False
            else:
                return self.loss < other.loss

    def is_dominate(self, other):
        # l_self = np.sum([np.linalg.norm(values.flatten()) for values in self.values])
        # l_other = np.sum([np.linalg.norm(values.flatten()) for values in other.values])

        if (
            self.l2 <= other.l2
            # self.l0 <= other.l0
            # self.num_perturbed <= other.num_perturbed
            and self.loss <= other.loss
            and (self.l2 < other.l2 or self.loss < other.loss)
            # and (self.l0 < other.l0 or self.loss < other.loss)
            # and (self.num_perturbed < other.num_perturbed or self.loss < other.loss)
        ):
            return 1
        elif (
            self.l2 >= other.l2
            # self.l0 >= other.l0
            # self.num_perturbed >= other.num_perturbed
            and self.loss >= other.loss
            and (self.l2 > other.l2 or self.loss > other.loss)
            # and (self.l0 > other.l0 or self.loss > other.loss)
            # and (self.num_perturbed > other.num_perturbed or self.loss > other.loss)
        ):
            return -1
        else:
            return 0


"""
params: {
  image,
  boxes,
  loss,
  population_size,
  pc,
  pm,
  pr0,
  perturbation_range,
  max_modified,
  tournament_size,
  max_archive_size,
  elite_prob (optinal)
}
"""


class POPOP:
    def __init__(self, params):
        self.params = params
        self.w = np.abs(params["boxes"][:, 2] - params["boxes"][:, 0])
        self.h = np.abs(params["boxes"][:, 3] - params["boxes"][:, 1])
        self.max_modified = np.array(
            [
                max(int(self.h[i] * self.w[i] * params["max_modified"]), 1)
                for i in range(self.w.shape[0])
            ]
        )
        self.locations = self.w * self.h
        self.p = np.array(
            [
                (1 - self.params["pr0"]) / 2,
                (1 - self.params["pr0"]) / 2,
                self.params["pr0"],
            ]
        )
        self.archive = []
        self.log = []

    def crossover1(self, a, b, max_modified):  # crossover for 1 object
        res = []
        n_var = a["pos"].shape[0]
        re = [(a, b), (b, a)]

        for r, e in re:
            M_r = r["pos"]
            assert M_r.shape[0] == np.unique(M_r).shape[0]
            M_e = e["pos"]
            # U = M_e \ (M_r \union M_e)
            # In other words, U is the set of pixel in only e
            U = np.setdiff1d(M_e, M_r)
            sizeAB = int(min(self.params["pc"] * max_modified, U.shape[0]))
            A = np.random.choice(M_r, sizeAB, replace=False)
            B = np.random.choice(U, sizeAB, replace=False)
            offspring = {"pos": [], "values": []}
            for i in range(n_var):
                if r["pos"][i] not in A:
                    offspring["pos"].append(r["pos"][i])
                    offspring["values"].append(r["values"][i])
                if e["pos"][i] in B:
                    offspring["pos"].append(e["pos"][i])
                    offspring["values"].append(e["values"][i])

            # offspring = Individual(np.array(offspring['pos']), np.array(offspring['values']))
            offspring["pos"] = np.array(offspring["pos"])
            offspring["values"] = np.array(offspring["values"])
            res.append(offspring)

        return res

    def mutation1(
        self, a, max_modified, elite_prob=0.5, obj_idx=None
    ):  # mutation for 1 object
        prob = (
            np.random.random()
        )  # decide whether to choose an elite in archive for mutation or generate a new one

        M_a = a["pos"]

        # if choosing an elite
        if prob < elite_prob and len(self.archive) > 0:
            scores = [ind.score_list[obj_idx] for ind in self.archive]
            top_scores_idxes = np.argsort(scores)[: max(len(self.archive) // 2, 1)]
            elite = random.choice(
                [self.archive[idx] for idx in top_scores_idxes]
            )  # Choosing a random top lowest fitness elite in archive
            # elite = min(self.archive) # Choosing the elite with lowest fitness or is_adv so far
            M_b = elite.pos[obj_idx]
            T = np.setdiff1d(M_b, M_a)
            sizeAB = int(min(self.params["pm"] * max_modified, T.shape[0]))
            A = np.random.choice(M_a, sizeAB, replace=False)
            B = np.random.choice(T, sizeAB, replace=False)

            new_a = {"pos": [], "values": []}
            for i in range(a["pos"].shape[0]):
                if a["pos"][i] not in A:
                    new_a["pos"].append(a["pos"][i])
                    new_a["values"].append(a["values"][i])
                if elite.pos[obj_idx][i] in B:
                    new_a["pos"].append(elite.pos[obj_idx][i])
                    new_a["values"].append(elite.values[obj_idx][i])

            # new_a = Individual(np.array(new_a['pos']), np.array(new_a['values']))
            new_a["pos"] = np.array(new_a["pos"])
            new_a["values"] = np.array(new_a["values"])
            return new_a

        # Otherwise, generating a new one for mutation
        T = np.setdiff1d(np.arange(self.locations[obj_idx]), M_a)
        sizeAB = int(min(self.params["pm"] * max_modified, T.shape[0]))
        A = np.random.choice(M_a, sizeAB, replace=False)
        B = np.random.choice(T, sizeAB, replace=False)
        new_a = {"pos": [], "values": []}
        for i in range(len(a["pos"])):
            if a["pos"][i] not in A:
                new_a["pos"].append(a["pos"][i])
                new_a["values"].append(a["values"][i])

        value_B = (
            np.random.choice([-1, 1, 0], size=(B.shape[0], 3), p=self.p)
            * self.params["perturbation_range"]
        )
        for i in range(B.shape[0]):
            new_a["pos"].append(B[i])
            new_a["values"].append(value_B[i])

        new_a["pos"] = np.array(new_a["pos"])
        new_a["values"] = np.array(new_a["values"])
        # new_a = Individual(np.array(new_a['pos']), np.array(new_a['values']))
        return new_a

    def tournament_selection(self, po, replacement = True):
        assert 1 <= self.params['tournament_size']
        assert self.params['population_size'] >= self.params['tournament_size']
#         if replacement:
#           assert len(po) >= self.params['tournament_size'] * self.params['population_size']

        new_parents = []
        while len(new_parents) < self.params['population_size']:
          pool = po.copy()
          random.shuffle(pool)
          for i in range(len(pool) // self.params['tournament_size']):
            best = None
            for j in range(self.params['tournament_size'] * i, self.params['tournament_size'] * (i + 1)):
#               print(j)
              if best is None or pool[j] < best:
                best = pool[j]
            new_parents.append(best)
            if len(new_parents) == self.params['population_size']:
              break
        return new_parents

    def run(self, n_gen=1000):
        # parents = [Individual(np.random.choice(self.locations, size = self.params['max_modified'], replace = False),
        #                       np.random.choice([-1, 1, 0], size = (self.params['max_modified'], 3), p=self.p) * self.params['perturbation_range'])
        #           for _ in range(self.params['population_size'])]
        parents = []
        for _ in range(self.params["population_size"]):
            pos = []
            values = []
            for i, max_modified in enumerate(self.max_modified):
                pos.append(
                    np.random.choice(
                        self.locations[i], size=max_modified, replace=False
                    )
                )
                values.append(
                    np.random.choice([-1, 1, 0], size=(max_modified, 3), p=self.p)
                    * self.params["perturbation_range"]
                )
            # pos = np.array(pos)
            # values = np.array(values)
            parents.append(Individual(pos, values))

        prev_conf = np.zeros(len(self.params["boxes"]))

        for parent in parents:
            parent.cal_fitness(self.params, prev_conf)

        self.append_archive(parents)

        pool = parents.copy()
        prev_conf = np.array([min([ind.f_score_list[idx] for ind in pool])
                                  for idx in range(len(self.params["boxes"]))])

        self.iters = 0
        for i in range(n_gen):
            if self.params["early_stop"] == 1:
                found = np.array([p.is_adv for p in pool])
                if np.any(found == True):
                    break
            
            if i == (n_gen - 1):
                break

            random.shuffle(pool)
            # Generate offsprings
            offsprings1 = [] # Generated by usual crossover and mutation processes
            for j in range(0, len(pool), 2):
                offsprings1.extend(self.crossover(pool[j], pool[j + 1]))

            offsprings1 = [
                self.mutation(o, self.params["elite_prob"]) for o in offsprings1
            ]
            for o in offsprings1:
                o.cal_fitness(self.params, prev_conf) # need addition val that is prev_conf
            pool.extend(offsprings1) # pool_size = 2*pop_size

            # offsprings2 = self.rearrange(pool) # Generated by rearranging each (pos, values) pair corresponding to each object in parents' individuals based on their scores
            # for o in offsprings2:
            #     o.cal_fitness(self.params, prev_conf) # with this, the number of evaluations in each generation increases by 2*pop_size
            # pool.extend(offsprings2) # pool_size = 4*pop_size => tournament_size = 8?

            # Tournament selection to keep good parents for next generation
            pool = self.tournament_selection(pool, replacement=True)

            self.append_archive(offsprings1)
            # self.append_archive(offsprings2)
            self.log.append([ind.loss for ind in pool])

            prev_conf = np.array([min([ind.f_score_list[idx] for ind in pool])
                                  for idx in range(len(self.params["boxes"]))])
            self.iters += 1

            """
            select the lowest conf_score (with sign) among all individuals corresponding to each object, assign to prev_conf
            """

        pool.sort()
        return pool[0]

    def append_archive(self, offsprings):
        for o in offsprings:
            if len(self.archive) == 0:
                self.archive.append(o)
                continue

            if o in self.archive:
                continue

            dominated = []
            be_dominated = False
            for elite in self.archive:
                comp = o.is_dominate(elite)
                if comp == 1:
                    dominated.append(elite)
                elif comp == -1:
                    be_dominated = True

            if len(dominated) > 0:
                for d in dominated:
                    self.archive.remove(d)
                self.archive.append(o)
            else:
                if len(self.archive) < self.params["max_archive_size"] and not (be_dominated):
                    self.archive.append(o)

    def crossover(self, a, b):  # crossover for multi-object
        # return self.crossover1(a, b, max_modified)
        A = copy.deepcopy(a)
        B = copy.deepcopy(b)

        for i in range(len(A.pos)):
            temp_a = {"pos": A.pos[i].copy(), "values": A.values[i].copy()}
            temp_b = {"pos": B.pos[i].copy(), "values": B.values[i].copy()}
            res = self.crossover1(temp_a, temp_b, self.max_modified[i])

            A.pos[i], B.pos[i] = res[0]["pos"], res[1]["pos"]
            A.values[i], B.values[i] = res[0]["values"], res[1]["values"]

        return [A, B]

    def mutation(self, a, elite_prob=0.5):  # mutation for multi-object
        # return self.mutation1(a, max_modified, elite_prob)
        A = copy.deepcopy(a)
        for i in range(len(A.pos)):
            temp_a = {"pos": A.pos[i].copy(), "values": A.values[i].copy()}
            res = self.mutation1(temp_a, self.max_modified[i], elite_prob, obj_idx=i)

            A.pos[i], A.values[i] = res["pos"], res["values"]

        return A

    def rearrange(self, parents):
        pop = copy.deepcopy(parents)
        for i in range(len(self.params["boxes"])):
            scores = np.array([ind.score_list[i] for ind in pop])
            adver_list = np.array([ind.adver_list[i] for ind in pop])
            top_scores_idxes = np.argsort(scores)
            pos_list = [copy.deepcopy(ind.pos[i]) for ind in pop]
            values_list = [copy.deepcopy(ind.values[i]) for ind in pop]
            f_score_list = [copy.deepcopy(ind.f_score_list[i]) for ind in pop]

            for j in range(len(top_scores_idxes)):
                pop[j].pos[i] = pos_list[top_scores_idxes[j]]
                pop[j].values[i] = values_list[top_scores_idxes[j]]
                pop[j].score_list[i] = scores[top_scores_idxes[j]]
                pop[j].adver_list[i] = adver_list[top_scores_idxes[j]]
                pop[j].f_score_list[i] = f_score_list[top_scores_idxes[j]]

        for ind in pop:
            # Can only do this if all the boxes doesn't overlap each other
            ind.loss = np.sum(ind.score_list)
            ind.is_adv = np.any(ind.adver_list)
            ind.cal_norm(self.params)

        return pop