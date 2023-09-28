import collections
import math
from json import dumps
from typing import List, Optional, Tuple, Union

import interegular
import torch
from pydantic import BaseModel

from outlines.text.generate.continuation import Continuation
from outlines.text.json_schema import build_regex_from_schema
from outlines.text.parsing import find_partial_matches, map_partial_states_to_vocab

from outlines.ab.utils.algorithms import *
from outlines.ab.utils.utils import *

# an actual constraining mechanism that isn't extremely slow
class LTL(Continuation):
    # uses linear-temporal logic to systematically constrain the generation 
    # of the language model. extremely cool. 

    # ideally, we could input a *set* of LTL formulas
    def __init__(self, model, ltl_lst: [], max_tokens: Optional[int]):
        super().__init__(model, max_tokens)

        vocabulary = model.tokenizer.vocabulary
        sorted_vocabulary = [
            model.tokenizer.convert_token_to_string(k)
            # this is probably really slow..
            for k, v in sorted(vocabulary.items(), key=lambda kv: kv[1])
        ]
        
        # get construct a buchi automata for each LTL property
        # LTL to BA is a fast conversion, especially if we convert to a nondeterministic buchi automata,
        # which, because of our smart sexy pre-processing, is fine
        rstr_set = set()
        for ltl in ltl_lst : rstr_set.add(ltl_to_ba(ltl))
        # print(self.rstr_set)

        self.distance_mappings = {}
        self.cstr_buchi = []
        self.current_state = [None] * len(ltl_lst)

        c = 0
        for buchi in rstr_set:
            """
            we go through and run pre-processing on each buchi automata to properly 
            prepare it for generation

            steps:
            1. reduce transitions by 
                - reducing all negative transitions to a "sigma" transition (e.g., takes the whole language)
                - clearing other negatives (or, maybe not? subtract from the vocab.)
                - removing transitions with multiple affirmative labels
            2. reduce accepting states to only states that have a self-loop sigma transition
            3. do the self-propagating reverse label search to give all nodes distance values

            some problems: because I have explicit language definitions and whatnot, having an arbitrary language
                           is a little weird and can cause problems. need to think of solution. flex-buchi?
            """
            
            """
            1, 2
            """
            n_usable_transitions = []
            n_acceptance_states = set()
            n_alphabet = {"Sigma*"}
            n_initial_state = buchi.initial_state
            n_states = set()
            for (sA, i, sB) in buchi.transitions:
                n_states.add(sA)
                n_states.add(sB)
                chars = i.split(" & ")
                num_full, num_neg = 0, 0
                for char in chars:
                    if char[0] == "!" : num_neg+=1
                    else : num_full +=1

                if num_full > 1 : continue
                elif num_full == 0 and num_neg >= 1 and sA == sB:
                    n_usable_transitions.append((sA, "Sigma*", sB))
                    n_acceptance_states.add(sA)
                # elif num_full == 1 and num_neg >= 1:
                #    usable_transitions.append()
                else:
                    n_alphabet.add(i)
                    n_usable_transitions.append((sA, i, sB))

            ltl_buchi = Buchi(n_states, n_initial_state, n_acceptance_states, n_alphabet, n_usable_transitions)

            """
            3. 
            similar to 3676 in ab.utils.algorithms

            2.5. simultanious BFS: establish distance between any state and the acceptance cycle
                2.5.1. start by assigning all acceptance states a distance value equal to the minimum acceptance cycle distance
                2.5.2. from each acceptance state, calculate the distance in waves, e.g., distance[new_state] = distance[state] + 1
                2.5.3. we only throw a new state on the stack if it is at least two less than the new state
                2.5.4. repeat until stack is empty (need to prove things about this algorithm...)
            """

            states = ltl_buchi.states
            initial_state = ltl_buchi.initial_state
            acceptance_states = ltl_buchi.acceptance_states

            distance = {}

            qx = []
            for state in acceptance_states: 
                distance[state] = 0
                qx.append(state)

            inf_states = set(states).difference(set(acceptance_states))
            for state in inf_states : distance[state] = float('inf')

            while qx:
                curr_state = qx.pop(0)
                curr_dist = distance[curr_state]
                rev_reachable = ltl_buchi.reachability_object_reverse[curr_state]

                for (r_state, char) in rev_reachable:
                    r_dist = distance[r_state]

                    if curr_dist + 1 < r_dist:
                        distance[r_state] = curr_dist + 1
                        qx.append(r_state)

            self.distance_mappings[ltl_buchi] = distance
            ltl_buchi.show_diagram()
            self.cstr_buchi.append(ltl_buchi)
            self.current_state[c] = ltl_buchi.initial_state

            c+=1

    # get the determined words within the language of the next state
    def next_state(self, ltl, c_state):
        distance = self.distance_mappings[ltl]

        # if we're already in the acceptance state, we *know* by construction
        # that we can stay in this state
        if distance[c_state] == 0 : return (c_state, "Sigma*")

        # go through all reachable states, find the state with the 
        # lowest distance to an accepting state, and go with that
        states = ltl.reachability_object[c_state]
        if states == set() : return None
        mm = float('inf')
        cl = None
        for state, char in states:
            if distance[state] < mm: 
                mm = distance[state]
                cl = (state, char)

        # if we don't have a state to transition to, sucks, we backed ourselves into 
        # a corner and can no longer generate.
        # additional work will have to be done to determine if we have intersecting LTL properties
        if mm == float('inf') : return None

        chars = cl[1].split(" & ")

        ct = ""
        for char in chars:
            if char[0] == "!" : ct+=char + ","
            else: 
                return (cl[0], char)

        return (cl[0], ct)

    def create_proposal(
        self, generated_token_ids: torch.LongTensor, logits: torch.DoubleTensor
    ) -> torch.DoubleTensor:
        """
        ok, here's what we want to do
        steps:
        1. get all adjacent nodes from the current node via the LTL reachability objects 
        2. using the distance metric from the pre-processing, we can greedily select a node to transition to next
            -> because any choice will eventually get us to an acceptance state, any choice is good lmao.
        3. if an LTL prop is not in an acceptance state with a sigma transition, we *require* next transition 
           be one that takes us closer to acceptance. chances are, we might miss some intersection emptiness thing
        4. limit 
        """

def ltl(model, ltl_array: [], max_tokens: Optional[int] = None):
    return LTL(model, ltl_array, max_tokens)


