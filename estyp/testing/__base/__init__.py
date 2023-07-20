
class TestResults:
    def __init__(self, res: dict, names: dict):
        self.__dict__.update(res)
        self.__names = names

    def __repr__(self):
        p_value = self.p_value
        if p_value < 0.0001:
            p_value = "<0.0001"
        else:
            p_value = f"{p_value:0.4f}"
        string = f"""
    {self.method}
    {len(self.method) * "-"}
    {self.__names['statistic']} = {self.statistic:0.4f} | df: {self.df} | p-value = {p_value}
    alternative hypothesis: {self.__names["alternative"]}"""
        if self.__dict__.get("conf_int"):
            cl = self.conf_level * 100
            if cl == float(int(cl)):
                cl = int(cl)
            string += f"""
    {cl} percent confidence interval:
    {" "}{self.conf_int[0]:0.6f} {self.conf_int[1]:0.6f}"""
        string += f"""
    sample estimates:
    {" " * 2}{self.__names["estimate"]}: {self.estimate:0.6f}
    """
        return string