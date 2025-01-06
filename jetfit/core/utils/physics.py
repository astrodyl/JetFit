from enum import Enum

from jetfit.core.defns.enums import TimeUnits, FluxUnits


class FluxConversions:
    """

    """
    cgs_mjy = 1.0e-26

    def __init__(self):
        pass

    @staticmethod
    def to_mjy(v: float, f: FluxUnits) -> float:
        """
        Converts the flux value to units of ``FluxUnits.MJY``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        Returns
        -------
        float
            The time value with units of ``FluxUnits.MJY``.
        """
        if f == FluxUnits.MJY:
            return v

        if f == FluxUnits.CGS:
            return v / FluxConversions.cgs_mjy

    @staticmethod
    def to_cgs(v: float, f: FluxUnits) -> float:
        """
        Converts the flux value to units of ``FluxUnits.CGS``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        Returns
        -------
        float
            The time value with units of ``FluxUnits.CGS``.
        """
        if f == FluxUnits.CGS:
            return v

        if f == FluxUnits.MJY:
            return v * FluxConversions.cgs_mjy

    @staticmethod
    def convert_to(v: float, f: FluxUnits | Enum, t: FluxUnits | Enum) -> float:
        """
        Converts the flux, ``v``, to ``t`` from ``f``.

        Parameters
        ----------
        v : float
            The flux value to convert.

        f : TimeUnits
            The units to convert from.

        t : TimeUnits
            The units to convert to.

        Returns
        -------
        float
            The flux value with units of ``t``.
        """
        if t == FluxUnits.MJY:
            return FluxConversions.to_mjy(v, f)

        if t == FluxUnits.CGS:
            return FluxConversions.to_cgs(v, f)

class TimeConversions:
    """

    """
    def __init__(self):
        pass

    @staticmethod
    def to_seconds(v: float, f: TimeUnits | Enum) -> float:
        """
        Converts the time value to units of ``TimeUnits.SEC``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        Returns
        -------
        float
            The time value with units of ``TimeUnits.SEC``.
        """
        if f == TimeUnits.SEC:
            return v

        if f == TimeUnits.HRS:
            return v * 24

        if f == TimeUnits.DAY:
            return v * 86400

    @staticmethod
    def to_hours(v: float, f: TimeUnits | Enum) -> float:
        """
        Converts the time value to units of ``TimeUnits.HRS``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        Returns
        -------
        float
            The time value with units of ``TimeUnits.HRS``.
        """
        if f == TimeUnits.HRS:
            return v

        if f == TimeUnits.SEC:
            return v / 3600

        if f == TimeUnits.DAY:
            return v * 24

    @staticmethod
    def to_days(v: float, f: TimeUnits | Enum) -> float:
        """
        Converts the time value to units of ``TimeUnits.DAY``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        Returns
        -------
        float
            The time value with units of ``TimeUnits.DAY``.
        """
        if f == TimeUnits.DAY:
            return v

        if f == TimeUnits.SEC:
            return v / 86400

        if f == TimeUnits.HRS:
            return v / 24

    @staticmethod
    def convert_to(v: float, f: TimeUnits | Enum, t: TimeUnits | Enum) -> float:
        """
        Converts the time value to ``t`` from ``f``.

        Parameters
        ----------
        v : float
            The time value to convert.

        f : TimeUnits
            The units to convert from.

        t : TimeUnits
            The units to convert to.

        Returns
        -------
        float
            The time value with units of ``TimeUnits.DAY``.
        """
        if t == TimeUnits.SEC:
            return TimeConversions.to_seconds(v, f)

        if t == TimeUnits.HRS:
            return TimeConversions.to_hours(v, f)

        if t == TimeUnits.DAY:
            return TimeConversions.to_days(v, f)
