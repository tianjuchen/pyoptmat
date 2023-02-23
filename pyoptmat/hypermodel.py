class HyperelasticModel(nn.Module):
    """
    This object provides the standard strain-based rate form of a constitutive model

    .. math::

      \\dot{\\sigma} = E \\left(\\dot{\\varepsilon} - (1-d) \\dot{\\varepsilon}_{in} \\right)

    Args:
      E:                        Material Young's modulus
      flowrule:                 :py:class:`pyoptmat.flowrules.FlowRule` defining the inelastic
                                strain rate
      dmodel (optional):        :py:class:`pyoptmat.damage.DamageModel` defining the damage variable
                                  evolution rate, defaults to :py:class:`pyoptmat.damage.NoDamage`
    """

    def __init__(self, C, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.C = C

    def forward(self, t, y, erate, T):
        """
        Return the rate equations for the strain-based version of the model

        Args:
          t:              (nbatch,) times
          y:              (nbatch,) stress
          erate:          (nbatch,) strain rates
          T:              (nbatch,) temperatures

        Returns:
          y_dot:          (nbatch,1) state rate
          d_y_dot_d_erate:(nbatch,1+nhist+1) Jacobian wrt the strain rate
        """
        stress = y[:, 0].clone()

        result = torch.empty_like(y, device=y.device)
        dresult = torch.zeros(y.shape + y.shape[1:], device=y.device)
        drate = torch.zeros_like(y)

        it = (
            -108 * self.C(T) ** 3
            + 6
            * 6 ** (0.5)
            * (54 * self.C(T) ** 6 + self.C(T) ** 3 * stress**3) ** (1 / 2)
            - stress**3
        ) ** (1 / 3)

        strain = (
            1
            / 6
            * (
                -it / self.C(T)
                - stress**2 / (self.C(T) * it)
                + stress / self.C(T)
                + 6
            )
        )

        temp1 = stress**2 * (
            3
            * self.C(T) ** 3
            * (9 * self.C(T) ** 6 + self.C(T) ** 3 * stress**3 / 6) ** (1 / 2)
            - 1
        )
        temp2 = (
            6 * 6 ** (0.5) * (54 * self.C(T) ** 3 + stress**3) ** (0.5)
            - 108 * self.C(T) ** 3
            - stress**3
        ) ** (2 / 3)

        temp = temp1 / temp2
        
        dstraindy = (
            1
            / (6 * self.C(T))
            * (-temp - (2 * stress / it - stress**2 * it ** (-2) * temp) + 1)
            + 1
        )
        result[:, 0] = 2 * erate * self.C(T) * (2*(strain-1)**(1/3) + 1)
        dresult[:, 0, 0] = 4/3 * self.C(T) * erate * (strain-1)**(-2/3) * dstraindy
        drate[:, 0] = 2 * self.C(T) * (2*(strain-1)**(1/3) + 1)
        Trate = torch.zeros_like(y)
        
        return result, dresult, drate, Trate
