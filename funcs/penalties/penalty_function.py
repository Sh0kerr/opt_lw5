def penalty_function(
        dot: tuple[float, float], 
        equalities: list=None, inequalities: list=None, 
        p: int=2
    ) -> float:

    sum_equalities = 0
    if equalities is not None:
        sum_equalities = sum(
            [
                abs(equality(dot))**p
                for equality in equalities
            ]
        )

    sum_inequalities = 0
    if inequalities is not None:
        sum_inequalities = sum(
            [
                max(0, inequality(dot))**p
                for inequality in inequalities
            ]
        )

    res = sum_equalities + sum_inequalities
    return res
