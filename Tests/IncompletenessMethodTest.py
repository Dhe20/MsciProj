import numpy as np
from Components.EventGenerator import EventGenerator
from Components.Inference import Inference

Gen = EventGenerator(dimension = 3, size = 50, sample_time=0.05*10**(-2), event_rate=10**3,
                     luminosity_gen_type = "Full-Schechter", coord_gen_type = "Random",
                     cluster_coeff=5, characteristic_luminosity=.5, total_luminosity=100,
                     event_distribution="Proportional", contour_type = "BVM", redshift_noise_sigma = 0,
                     resolution=200, plot_contours=True, seed = 10)

Data = Gen.GetSurveyAndEventData(min_flux=2*np.min(Gen.fluxes))
I = Inference(Data, survey_type='perfect', resolution_H_0=100)


def H_0_inference_3d_perfect_survey_vectorised_incomplete():
    H_0_recip = np.reciprocal(I.H_0_range)[:, np.newaxis]

    redshifts = np.tile(I.SurveyAndEventData.detected_redshifts, (I.resolution_H_0, 1))

    Ds = I.c * redshifts * H_0_recip

    # In Survey Terms (above Fth)

    P_G = 1

    burr_full = I.get_vectorised_burr(Ds)
    vmf = I.get_vectorised_vmf()
    luminosity_term = I.lum_term[I.event_distribution_inf](redshifts)

    P_GWdata_given_G = np.sum(burr_full * vmf * luminosity_term, axis=2) * P_G

    if I.p_det:
        p_det_vec_given_G = luminosity_term * I.get_p_det_vec(Ds)  # Change the final term
        P_det_total_given_G = np.sum(p_det_vec_given_G, axis=1)
        # Note it is the same for all galaxies - collapse the cube first
        P_GWdata_given_G = np.divide(P_GWdata_given_G, P_det_total_given_G)

    # Out of Survey Terms (below Fth)

    P_Gbar = 1 - P_G

    # This should be dealt with after the galaxy side of the cube is collapsed

    P_GWdata_given_Gbar = 1 #Replace this term

    P_GWdata_given_Gbar = P_GWdata_given_Gbar*P_Gbar

    if I.p_det:
        p_det_vec_given_Gbar = luminosity_term * I.get_p_det_vec(Ds)  # Change the final term
        P_det_total_given_Gbar = np.sum(p_det_vec_given_Gbar, axis=1)
        # Note it is the same for all galaxies - collapse the cube first
        P_GWdata_given_Gbar = np.divide(P_GWdata_given_Gbar, P_det_total_given_Gbar)

    likelihood_expression = P_GWdata_given_G+P_GWdata_given_Gbar

    H_0_pdf_single_event = likelihood_expression

    H_0_pdf = np.product(H_0_pdf_single_event, axis=0)

    H_0_pdf /= np.sum(H_0_pdf) * (I.H_0_increment)
    return H_0_pdf

H_0_inference_3d_perfect_survey_vectorised_incomplete()