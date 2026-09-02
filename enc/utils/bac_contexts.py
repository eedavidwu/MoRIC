
import math

import torch
from enc.component.coolchic import _laplace_cdf
from enc.utils.misc import bac_state_idx_from_proba_0

N_MUQ = 16
N_SIGQ = 50

SIG_LOG_MIN = -5 + 4
SIG_LOG_MAX_EXCL = 5 + 4


P_MIN = torch.tensor([0.001])
P_MAX = torch.tensor([1 - 0.001])


def reasonable_proba(p):
    p = torch.abs(p)
    if p < P_MIN:
        p = P_MIN
    if p > P_MAX:
        p = P_MAX
    return p


def get_contexts(contexts_cpp: str = ""):
    inputs = torch.arange(
        SIG_LOG_MIN, SIG_LOG_MAX_EXCL, (SIG_LOG_MAX_EXCL - SIG_LOG_MIN) / N_SIGQ
    )
    sigs_quanted = torch.exp(inputs - 4).to("cpu")

    probas = []

    R = torch.tensor([0])
    mu_min = 0 - N_MUQ // 2
    mu_max = (
        N_MUQ // 2 + 1
    )

    for mu_offset in range(mu_min, mu_max):
        mu_offset = torch.tensor([mu_offset])
        sigs = []
        for sig in sigs_quanted:
            gt0_surface = _laplace_cdf(R + 0.5, mu_offset / N_MUQ, sig) - _laplace_cdf(
                R - 0.5, mu_offset / N_MUQ, sig
            )
            gt0 = gt0_surface / 1.0
            gt0 = reasonable_proba(gt0)
            if gt0 == P_MAX:
                gt1 = torch.tensor([0.5])
                gt2 = torch.tensor([0.5])
                gt3 = torch.tensor([0.5])
            else:
                gt1_surface = (
                    _laplace_cdf(R + 1 + 0.5, mu_offset / N_MUQ, sig)
                    - _laplace_cdf(R + 1 - 0.5, mu_offset / N_MUQ, sig)
                ) + (
                    _laplace_cdf(R - 1 + 0.5, mu_offset / N_MUQ, sig)
                    - _laplace_cdf(R - 1 - 0.5, mu_offset / N_MUQ, sig)
                )
                if gt1_surface <= P_MIN:
                    gt1 = torch.tensor([0.5])
                    gt2 = torch.tensor([0.5])
                    gt3 = torch.tensor([0.5])
                else:
                    gt1 = gt1_surface / (1 - gt0_surface)
                    gt1 = reasonable_proba(gt1)
                    gt2_surface = (
                        _laplace_cdf(R + 2 + 0.5, mu_offset / N_MUQ, sig)
                        - _laplace_cdf(R + 2 - 0.5, mu_offset / N_MUQ, sig)
                    ) + (
                        _laplace_cdf(R - 2 + 0.5, mu_offset / N_MUQ, sig)
                        - _laplace_cdf(R - 2 - 0.5, mu_offset / N_MUQ, sig)
                    )
                    if gt2_surface <= P_MIN:
                        gt2 = torch.tensor([0.5])
                        gt3 = torch.tensor([0.5])
                    else:
                        gt2 = gt2_surface / (1 - gt0_surface - gt1_surface)
                        gt2 = reasonable_proba(gt2)
                        gt3_surface = (
                            _laplace_cdf(R + 3 + 0.5, mu_offset / N_MUQ, sig)
                            - _laplace_cdf(R + 3 - 0.5, mu_offset / N_MUQ, sig)
                        ) + (
                            _laplace_cdf(R - 3 + 0.5, mu_offset / N_MUQ, sig)
                            - _laplace_cdf(R - 3 - 0.5, mu_offset / N_MUQ, sig)
                        )
                        if gt3_surface <= P_MIN:
                            gt3 = torch.tensor([0.5])
                        else:
                            gt3 = gt3_surface / (
                                1 - gt0_surface - gt1_surface - gt2_surface
                            )
                            gt3 = reasonable_proba(gt3)

            pos_surface = 1.0 - _laplace_cdf(R + 0.5, mu_offset / N_MUQ, sig)
            neg_surface = _laplace_cdf(R - 0.5, mu_offset / N_MUQ, sig)
            if pos_surface <= P_MIN and neg_surface <= P_MIN:
                ppos = torch.tensor([0.5])
            elif pos_surface <= P_MIN:
                ppos = torch.tensor([0])
            elif neg_surface <= P_MIN:
                ppos = torch.tensor([1])
            else:
                ppos = pos_surface / (pos_surface + neg_surface)
            ppos = reasonable_proba(ppos)

            these_probas = {
                "gt0": gt0,
                "gt1": gt1,
                "gt2": gt2,
                "gt3": gt3,
                "ppos": ppos,
            }
            if (
                math.isnan(gt0)
                or math.isnan(gt1)
                or math.isnan(gt2)
                or math.isnan(gt3)
                or math.isnan(ppos)
                or gt0 < 0
                or gt1 < 0
                or gt2 < 0
                or gt3 < 0
                or ppos < 0
            ):
                print("NAN in table!")
                print("mu_offset", mu_offset, "sig", sig, "idx", len(probas))
                print(these_probas)
                exit(1)
            sigs.append(these_probas)
        probas.append(sigs)

    contexts = []
    for sigs in probas:
        sig_ctxs = []
        for ps in sigs:
            gt0 = bac_state_idx_from_proba_0(ps["gt0"])
            gt1 = bac_state_idx_from_proba_0(ps["gt1"])
            gt2 = bac_state_idx_from_proba_0(ps["gt2"])
            gt3 = bac_state_idx_from_proba_0(ps["gt3"])
            ppos = bac_state_idx_from_proba_0(ps["ppos"])
            these_ctxs = {"gt0": gt0, "gt1": gt1, "gt2": gt2, "gt3": gt3, "ppos": ppos}
            sig_ctxs.append(these_ctxs)
        contexts.append(sig_ctxs)

    if contexts_cpp != "":
        with open(contexts_cpp + ".h", "wt") as f:
            print(
                f"""


// some numbers and indices related to mu and sig quantization.
int const N_MUQ = {N_MUQ};  // number of mu offsets.
int const N_SIGQ = {N_SIGQ}; // number of sig values. now 50, so multiple of 10
int const ZERO_MU = N_MUQ/2;

int const SIG_LOG_MIN = {SIG_LOG_MIN}; // this min is IN the set.
int const SIG_LOG_MAX_EXCL = {SIG_LOG_MAX_EXCL}; // this max is NOT in the set.

int const PROBA_50_STATE = (2*32+1); // generate a BinProbModel_Std with 50% probability.

inline
void get_val_mu_indicies(int val_mu, int val_log_sig,
                         int &r_val_mu_rounded, int &r_val_mu_index, int &r_val_log_sig_index)
{{
    int val_mu_rounded = val_mu;
    val_mu_rounded = (val_mu_rounded >= 0) ? (val_mu_rounded+ARM_SCALE/2)>>ARM_PRECISION<<ARM_PRECISION : -((-val_mu_rounded+ARM_SCALE/2)>>ARM_PRECISION<<ARM_PRECISION);

    int val_mu_index = (val_mu - val_mu_rounded)*N_MUQ;
    // round to an index
    val_mu_index = val_mu_index >= 0 ? ((val_mu_index+ARM_SCALE/2)>>ARM_PRECISION) : -((-val_mu_index+ARM_SCALE/2)>>ARM_PRECISION);
    val_mu_index += N_MUQ/2;

    // no longer a table.
    int val_log_sig_index;
    val_log_sig -= SIG_LOG_MIN*ARM_SCALE;
    if (val_log_sig < 0)
        val_log_sig_index = 0;
    else
    {{
        val_log_sig_index = val_log_sig*(N_SIGQ/(SIG_LOG_MAX_EXCL-SIG_LOG_MIN))+ARM_SCALE/2;
        val_log_sig_index >>= ARM_PRECISION;
        if (val_log_sig_index >= N_SIGQ)
            val_log_sig_index = N_SIGQ-1;
    }}

    r_val_mu_rounded = val_mu_rounded>>ARM_PRECISION;
    r_val_mu_index = val_mu_index;
    r_val_log_sig_index = val_log_sig_index;
}}


// contexts {len(contexts)} mus, {len(contexts[0])} sigmas
// Context numbers for gtx and ppos for a given mu and sigma.
class MuSigGTs
{{
public:
    MuSigGTs(int gt0, int gt1, int gt2, int gt3, int ppos)
    {{
        m_gt0 = BinProbModel_Std(gt0);
        m_gt1 = BinProbModel_Std(gt1);
        m_gt2 = BinProbModel_Std(gt2);
        m_gt3 = BinProbModel_Std(gt3);
        m_ppos = BinProbModel_Std(ppos);
    }}
    ~MuSigGTs() {{}}
public:
    BinProbModel_Std m_gt0;
    BinProbModel_Std m_gt1;
    BinProbModel_Std m_gt2;
    BinProbModel_Std m_gt3;
    BinProbModel_Std m_ppos;
}};

extern MuSigGTs g_contexts[N_MUQ+1][N_SIGQ];""",
                file=f,
            )

        with open(contexts_cpp + ".cpp", "wt") as f:
            print(
                f"""

#include "Contexts.h"
#include "common.h"
#include "cc-contexts.h"

MuSigGTs g_contexts[N_MUQ+1][N_SIGQ] = {{""",
                file=f,
            )

            for mu_idx in range(len(contexts)):
                print("{", file=f)
                for sig_idx in range(len(contexts[mu_idx])):
                    ctxs = contexts[mu_idx][sig_idx]
                    print(
                        "  MuSigGTs( %d,%d,%d,%d,%d ),"
                        % (
                            ctxs["gt0"],
                            ctxs["gt1"],
                            ctxs["gt2"],
                            ctxs["gt3"],
                            ctxs["ppos"],
                        ),
                        file=f,
                    )
                print("},", file=f)

            print("};", file=f)

        print(contexts_cpp + ".h", "created")
        print(contexts_cpp + ".cpp", "created")

    return contexts, sigs_quanted, probas


