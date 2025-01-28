"""The tracking tool aims to provide a CLI for two important tracking tools.

We can generate tracking data for a given run of PROCESS, which corresponds to one output MFile.
When generating tracking data, we store metadata about the run, such as time and commit message,
and also variable data: the value of a given variable during this run; this is all stored in one
JSON file.

* Run title: the name of the run input, and hence output, file (e.g. baseline 2018).
* Tracking history: the data held within all JSON files in the database that span many variables,
many different run titles and spans a period of time.
* Variable history: the data held within all JSON files in the database for a given variable;
not all run titles will have a history for each variable being tracked
* A run of PROCESS: refers to the output MFile from the running of one input file
* JSON file: holds the metadata, and tracking data, generated by a run of PROCESS

We can also plot a database of JSON files. The tracking history (ie all the data) must be loaded
before being processed into a dataframe. This dataframe is then processed into a tracking dashboard:

* The dashboard: represented by the entire html file generated, contains many panels
* Panels: each parent module (or variable namespace) has its own tab which contains many graphs
* Graphs: each graph shows the variable history of one variable, it will have many lines
* Lines: each line represents many datapoints for a single variable, over a period of time,
that all come from the same run title
* Datapoints: the value of a variable at a point in time during a given run


To add a variable to track:

Add the variable to ProcessTracker.tracking_variables (in this file).
If the variable is not a fortran module variable, ensure to override its parent module name
e.g. FOO.bar says `bar`'s parent module is `FOO`.
"""

import argparse
import datetime
import inspect
import itertools
import json
import logging
import math
import pathlib
from typing import ClassVar

import git
import pandas as pd
from bokeh.embed import file_html
from bokeh.layouts import gridplot
from bokeh.models import (
    ColumnDataSource,
    DatetimeTickFormatter,
    HoverTool,
    TabPanel,
    Tabs,
)
from bokeh.palettes import Category10
from bokeh.plotting import figure
from bokeh.resources import CDN

from process import fortran
from process.io import mfile as mf

logging.basicConfig(level=logging.INFO, filename="tracker.log")
logger = logging.getLogger("PROCESS Tracker")


# Tracking


class TrackingFile:
    """Acts as the data storage for a given JSON file ie. holds the data from a run of PROCESS

    e.g. starfire_MFILE-<date>-<time>.json
    """

    def __init__(self) -> None:
        self.meta: dict = {}
        # metadata (e.g. commit message and date generated) as a key-value pair

        self.tracking: dict = {}
        # tracking data that shows the value of an important variable as a key-value pair

    def asdict(self) -> dict:
        return {"meta": self.meta, "tracking": self.tracking}


class ProcessTracker:
    """Manages the creation of tracking data into a JSON file for a run of PROCESS."""

    meta_variables: ClassVar = {"date", "time"}
    # Variables in an MFile that hold metadata we want to show on the graph

    tracking_variables: ClassVar = {
        "pheat",
        "bootstrap_current_fraction",
        "pinjmw",
        "shldith",
        "fwith",
        "fwoth",
        "thshield",
        "tftsgap",
        "dr_bore",
        "dr_cs",
        "scrapli",
        "blnkoth",
        "precomp",
        "tfthko",
        "blnkith",
        "vvblgap",
        "scraplo",
        "gapoh",
        "gapsto",
        "shldoth",
        "tfcth",
        "gapds",
        "pnucshld",
        "pnucblkt",
        "triang",
        "triang95",
        "p_plasma_inner_rad_mw",
        "tesep",
        "f_nd_alpha_electron",
        "wallmw",
        "aspect",
        "rminor",
        "rmajor",
        "q95",
        "te",
        "beta",
        "inductive_current_fraction",
        "zeff",
        "bt",
        "hfact",
        "kappa",
        "fusion_power",
        "teped",
        "powerht",
        "kappa95",
        "neped",
        "dene",
        "p_plasma_rad_mw",
        "ne0",
        "aux_current_fraction",
        "nd_impurities",
        "taueff",
        "te0",
        "pdivt",
        "nesep",
        "vol_plasma",
        "a_plasma_surface",
        "pnetelmw",
        "etath",
        "pgrossmw",
        "tftmp",
        "n_tf",
        "bmaxtf",
        "vstot",
        "nd_ions_total",
        "t_burn",
        "divlife",
        "CostModel2.step20",
        "CostModel2.step21",
        "CostModel2.step2101",
        "CostModel2.step2202",
        "CostModel2.step2203",
        "CostModel2.step22",
        "CostModel2.step23",
        "CostModel2.step24",
        "CostModel2.step25",
        "cdirt",
        "concost",
    }
    # Variable in an MFile that we will track the changes in.

    # 1) A variable of the form X.Y shows that variable Y exists as a member of module Y. This is needed for variables no longer in Fortran modules, where their parent cannot be scraped.
    # 2) Other variables should be a module variable for a wrapped module. The name of the parent module will be automatically determined.

    def __init__(
        self,
        mfile: str,
        database: str | None = None,
        message: str | None = None,
        hashid: str | None = None,
    ) -> None:
        """Drive the creation of tracking JSON files.

        :param mfile: the path to an mfile to create tracking data for.
        :type mfile: str

        :param database: the folder (acting as a database) that stores all tracking JSON files
        :type database: str
        """
        self.mfile = mf.MFile(mfile)
        self.tracking_file = TrackingFile()

        self._generate_data()

        title = pathlib.Path(mfile).stem
        self.add_extra_metadata("title", title)

        commit_message = message or git.git_commit_message()
        self.add_extra_metadata("commit_message", commit_message)
        commit_hash = hashid or git.git_commit_hash()
        self.add_extra_metadata("commit_hash", commit_hash)

        if database:
            # split to remove timezone from date if present
            date = self.tracking_file.meta.get("date").replace("/", "").split(" ")[0]
            time = self.tracking_file.meta.get("time")

            # for an mfile called foo.MFILE.DAT created at 16:00 on 15/11/2021
            # the tracking data file will be written to
            # foo_MFILE-15112021-16:00.json
            tracking_file_name = pathlib.Path(database) / f"{title}-{date}-{time}.json"

            self.write(tracking_file_name)

    def add_extra_metadata(self, key, value):
        """Enables the adding of extra metadata of key: value"""
        self.tracking_file.meta[key] = value

    def _generate_data(self):
        """
        Generates metadata for all metadata variables in ProcessTracker.meta_variables
        Generates tracking data for all variables in ProcessTracker.tracking_variables.

        Extracts the various meta/variable data from the mfile into our internal
        data store
        """
        # meta data
        for var in self.meta_variables:
            try:
                # value of var in the mfile
                variable_data = self.mfile.data[var]
            except KeyError:
                logger.info(f"{var} is not present in the MFile and will be skipped.")
                continue

            self.tracking_file.meta[var] = variable_data.get_scan(1)

        # tracking data
        for var in self.tracking_variables:
            if "." in var:
                # a dotted variable is for variables that no longer exist in Fortran module variables
                # see tracking_variables docstring
                try:
                    _, var = var.split(".")
                except (AttributeError, ValueError):
                    logger.warning(
                        f"{var} is a dotted variable and must be in the form OVERRIDINGNAME.VARIABLE"
                    )

            # value of the variable extracted from the mfile
            mfile_var_value = self.mfile.data[var]

            if isinstance(mfile_var_value, mf.MFileErrorClass):
                logger.info(f"{var} is not present in the MFile and will be skipped.")
                continue

            if mfile_var_value.get_number_of_scans() > 1:
                logger.info(
                    f"Only scan 1 will be tracked, but {var} has {mfile_var_value.get_number_of_scans()} scans."
                )

            self.tracking_file.tracking[var] = mfile_var_value.get_scan(1)

    def write(self, filename: str):
        """Write the metadata and tracking data into the JSON file"""
        with open(filename, "w") as f:
            json.dump(self.tracking_file.asdict(), f, indent=4)


# Plotting

colour_map = {}
colours = itertools.cycle(
    Category10[10]
)  # hardcode at 10 meaning a maximum of 10 different line colours


def get_line_colour(title: str):
    """Given a run title, return either cached colour or pick a colour and then cache it."""
    if title in colour_map:
        return colour_map[title]

    colour = next(colours)
    colour_map[title] = colour

    return colour


class TrackedVariable:
    """
    Holds the variable history, over all runs that produced this variable in their output,
    of this variable.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        # Name of the graph this variable is plotted under
        self._data = []
        # Tuple of (title, timestamp, data, message, hash)

        # title: name of the run e.g. starfire or baseline_2018 (not unique)
        # timestamp: date of the specific run this tuple refers to (unique)
        # data: the value of the variable, `name`, on `title` run at time `date` (possibly unique)
        # message: latest commit message when `title` was run at time `date` (possibly unique)
        # hash: latest commit hash when `title` was run at time `date` (possibly unique)

    def add_datapoint(self, title, data, message, commit_hash, timestamp):
        """
        Adds a tuple to this graph

        title: name of the run e.g. starfire or baseline_2018 (not unique)
        timestamp: date of the specific run this tuple refers to (unique)
        data: the value of the variable, `name`, on `title` run at time `date` (possibly unique)
        message: latest commit message when `title` was run at time `date` (possibly unique)
        """
        self._data.append((title, timestamp, data, message, commit_hash))

    def as_dataframe(self):
        """
        Converts our internal data representation of this variable's history as a dataframe.

        title -> title
        timestamp -> date
        data -> value
        message -> annotation
        """
        df = pd.DataFrame(self._data)
        df.columns = ("title", "date", "value", "commit", "commit_id")

        return df


class TrackedData:
    """Holds the entire tracking history of a database"""

    def __init__(self, database) -> None:
        self.database = pathlib.Path(database)

        self.tracked_variables = {}
        # Each tracked variable corresponds to a TrackedVariable

        self._track()

    def _add_variables(self, json_file_data):
        """Adds the `data` of an entire JSON tracking
        file to an internal store before being
        transformed into a dataframe, to then be plotted.
        """

        # extract the metadata from our file as all datapoints of this file will require them
        metadata = json_file_data["meta"]
        title = metadata.get("title", "-")
        message = metadata.get("commit_message", "-")
        hash_ = metadata.get("commit_hash", "-")
        # split to disregard timezone if present
        date_str = metadata.get("date", "-").split(" ")[0]
        time_str = metadata.get("time", "-")

        # common format for the timestamp of the run
        data_time_str = f"{date_str.strip()} - {time_str.strip()}"
        date_time = datetime.datetime.strptime(
            data_time_str, "%d/%m/%Y - %H:%M"
        ).replace(tzinfo=datetime.timezone.utc)

        tracking_data = json_file_data.get(
            "tracking", []
        )  # the JSON data of one run of PROCESS in python datastructures
        for variable, value in tracking_data.items():
            # create a new TrackedVariable when we see a variable we do not know
            if variable not in self.tracked_variables:
                self.tracked_variables[variable] = TrackedVariable(variable)

            self.tracked_variables.get(variable).add_datapoint(
                title, value, message, hash_, date_time
            )

    def _track(self):
        """Loads the entire history, for all runs that are stored in the database."""
        # open all files in the `database` folder

        for i in self.database.glob("*.json"):
            with open(i) as f:
                file_data = json.load(f)  # parsed contents of the JSON tracking file
                self._add_variables(
                    file_data
                )  # add all the data in this JSON file to our internal store


def plot_tracking_data(database):
    """
    Drives the processing of existing .json tracking files and then plotting of this processed data.
    """
    loaded_tracking_database_data = TrackedData(database)
    # data is our entire database of JSON files loaded and processed

    figures = {}

    overrides = {}
    # holds a map of variable parent's names overriden using the . (dot) syntax
    # variable: parent module name

    # populates the overrides map
    for i in ProcessTracker.tracking_variables:
        if hasattr(i, "split") and len(name_and_variable := i.split(".")) == 2:
            overriden_name, variable = name_and_variable
            overrides[variable] = overriden_name

    for variable, history in loaded_tracking_database_data.tracked_variables.items():
        df = (
            history.as_dataframe()
        )  # all the data for one tracked variable as a dataframe

        # order by date to avoid polygons all over the plot
        df = df.sort_values("date", ascending=True)

        # overrides trumps fortran scrapping
        parent = overrides.get(
            variable
        ) or PythonFortranInterfaceVariables.parent_module(
            variable
        )  # module name (or given name if overridden)

        if not parent:
            logger.warning(
                f"Variable {variable} is not a module variable of any Fortran module, please provide the python class which this variable can be found under as CLASS.VARIABLE noting that VARIABLE must be a variable present in the output file and does not necessarily need to correspond to an actual class variable, VARIABLE, of CLASS"
            )
            continue

        if figures.get(parent) is None:
            figures[parent] = []

        titles = list(
            df["title"].to_numpy()
        )  # all scenarios this variable is tracked in

        figur = figure(title=variable, x_axis_type="datetime", width=600, height=600)
        figur.xaxis.formatter = DatetimeTickFormatter(
            hours="%d %B %Y",
            days="%d %B %Y",
            months="%d %B %Y",
            years="%d %B %Y",
        )
        figur.xaxis.major_label_orientation = math.pi / 4

        # each title (different scenario) has a different line colour
        for t in set(titles):
            run_title_dataframe = df[
                df["title"] == t
            ]  # the variable history for each (applicable) run title
            subsource = ColumnDataSource(
                run_title_dataframe
            )  # convert dataframe into Bokeh compatible
            colour = get_line_colour(t)

            figur.scatter(
                x="date", y="value", source=subsource, legend_label=t, color=colour
            )

            figur.line(
                x="date",
                y="value",
                source=subsource,
                legend_label=t,
                color=colour,
                line_color=colour,
            )

        figur.legend.click_policy = "hide"  # hide a line if its legend entry is clicked

        # show a title, commit, and date when hovering over a datapoint
        hovertool = HoverTool(
            tooltips=[
                ("Title", "@title"),
                ("Commit", "@commit"),
                ("Commit ID", "@commit_id"),
                ("Date", "@date{%Y-%m-%d %H:%M}"),
            ],
            formatters={
                "@title": "printf",
                "@date": "datetime",
                "@commit": "printf",
                "@commit_id": "printf",
            },
        )

        # each section of the legend is next to each other
        # in a line rather than the default stack
        figur.legend.orientation = "horizontal"

        # legend is positioned in the x center
        figur.legend.location = "center"

        # legend is above the actual figure
        # so as not to cover the data
        figur.add_layout(figur.legend[0], "above")

        figur.add_tools(hovertool)

        figures[parent].append(figur)

    panels = []

    # each module/overriden name e.g. CostModel2 has a panel which holds graphs for all variables under that scope
    for parent_module_name, figs in figures.items():
        if len(figs) % 2 != 0:
            gplot = gridplot([figs[i : i + 2] for i in range(0, len(figs), 2)])
        else:
            gplot = gridplot([figs[i : i + 2] for i in range(0, len(figs) - 1, 2)])

        panels.append(TabPanel(child=gplot, title=parent_module_name))

    tabs = Tabs(tabs=panels)

    # returns the entire HTML document (to be written to a file)
    return file_html(tabs, CDN, "PROCESS Regression Testing Visualisation")


def write_tracking_html_file(database, output):
    """Writes the visual tracking data to an appropriate file"""
    tracking_html = plot_tracking_data(database)

    with open(output, "w") as f:
        f.write(tracking_html)


class PythonFortranInterfaceVariables:
    """
    Parses the f2py generated tree and can find the parent module of a module variable.
    """

    tree: ClassVar = {}

    @classmethod
    def populate_classes(cls):
        """
        Scrape all the data into an internal data storage
        """
        classes = {}

        for name, module in inspect.getmembers(fortran):
            # there is technically a `fortran` type
            # that is not clear where it is held
            # this allow checking that module is a
            # fortran module and is the next best
            # thing to check this is a module.
            # physics_variables is just the chosen arbitrary module
            if type(module) == type(fortran.physics_variables):  # noqa: E721
                classes[name] = cls._get_variables(module)

        cls.tree = classes

    @classmethod
    def _get_variables(cls, fortran_module):
        """
        Get the variables of a given module
        """
        variables = []

        for name, function in inspect.getmembers(fortran_module):
            # type(fortran.physics_variables.init_physics_variables) => fortran subroutine
            # if its not a fortran subroutine, its a variable
            # because type `fortran` cannot be checked as a type
            if type(function) != type(fortran.physics_variables.init_physics_variables):  # noqa: E721
                variables.append(name)

        return variables

    @classmethod
    def parent_module(cls, var: str):
        """
        Return the parent module of var, or None.
        """
        if not cls.tree:
            cls.populate_classes()

        for mod, functions in cls.tree.items():
            if var in functions:
                return mod

        return None


def track_entrypoint(arguments):
    """
    Entrypoint if we run in track mode.

    Generates a tracking JSOn file for the provided MFile.
    """
    if not arguments.mfile:
        raise ValueError("track requires --mfile be set")

    ProcessTracker(
        mfile=arguments.mfile,
        database=arguments.db,
        message=arguments.commit,
        hashid=arguments.hash,
    )


def plot_entrypoint(arguments):
    """
    Entrypoint if we run in plot mode.

    Plots all tracking data into a single tracking.html file
    """
    if not arguments.out:
        raise ValueError("plot requires --out be set")

    write_tracking_html_file(database=arguments.db, output=arguments.out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("mode", type=str, choices=["track", "plot"])

    parser.add_argument("db", type=str)
    parser.add_argument("-o", "--out", type=str, default=None)
    parser.add_argument("-m", "--mfile", type=str, default=None)
    parser.add_argument(
        "--commit",
        type=str,
        default=None,
        help="The current commit message. If not provided, the code attempts to query to Git repository.",
    )
    parser.add_argument(
        "--hash",
        type=str,
        default=None,
        help="The current commit hash. If not provided, the code attempts to query to Git repository.",
    )

    arguments = parser.parse_args()

    if arguments.mode == "track":
        track_entrypoint(arguments)
    elif arguments.mode == "plot":
        plot_entrypoint(arguments)
