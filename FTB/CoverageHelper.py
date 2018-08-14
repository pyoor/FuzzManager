#!/usr/bin/env python
# encoding: utf-8
'''
CoverageHelper -- Various methods around processing coverage data

@author:     Christian Holler (:decoder)

@license:

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at http://mozilla.org/MPL/2.0/.

@contact:    choller@mozilla.com
'''

# Ensure print() compatibility with Python 3
from __future__ import print_function

import re


def merge_coverage_data(r, s):
    # These variables are mainly for debugging purposes. We count the number
    # of warnings we encounter during merging, which are mostly due to
    # bugs in GCOV. These statistics can be included in the report description
    # to track the status of these bugs.
    stats = {
        'null_coverable_count': 0,
        'length_mismatch_count': 0,
        'coverable_mismatch_count': 0
    }

    def merge_recursive(r, s):
        assert(r['name'] == s['name'])

        if "children" in s:
            for child in s['children']:
                if child in r['children']:
                    # Slow path, child is in both data blobs,
                    # perform recursive merge.
                    merge_recursive(r['children'][child], s['children'][child])
                else:
                    # Fast path, subtree only in merge source
                    r['children'][child] = s['children'][child]
        else:
            rc = r['coverage']
            sc = s['coverage']

            # GCOV bug, if the file has 0% coverage, then all of the file
            # is reported as not coverable. If s has that property, we simply
            # ignore it. If r has that property, we replace it by s.
            if sc.count(-1) == len(sc):
                if rc.count(-1) != len(rc):
                    #print("Warning: File %s reports no coverable lines" % r['name'])
                    stats['null_coverable_count'] += 1
                return

            if rc.count(-1) == len(rc):
                if sc.count(-1) != len(sc):
                    #print("Warning: File %s reports no coverable lines" % r['name'])
                    stats['null_coverable_count'] += 1

                r['coverage'] = sc
                return

            # grcov does not always output the correct length for files when they end in non-coverable lines.
            # We record this, then ignore the excess lines.
            if len(rc) != len(sc):
                #print("Warning: Length mismatch for file %s (%s vs. %s)" % (r['name'], len(rc), len(sc)))
                stats['length_mismatch_count'] += 1

            # Disable the assertion for now
            #assert(len(r['coverage']) == len(s['coverage']))

            minlen = min(len(rc), len(sc))

            for idx in range(0, minlen):
                # There are multiple situations where coverage reports might disagree
                # about which lines are coverable and which are not. Sometimes, GCOV
                # reports this wrong in headers, but it can also happen when mixing
                # Clang and GCOV reports. Clang seems to consider more lines as coverable
                # than GCOV.
                #
                # As a short-term solution we will always treat a location as *not* coverable
                # if any of the reports says it is not coverable. We will still record these
                # mismatches so we can track them and confirm them going down once we fix the
                # various root causes for this behavior.
                if (sc[idx] < 0 and rc[idx] >= 0) or (rc[idx] < 0 and sc[idx] >= 0):
                    #print("Warning: Coverable/Non-Coverable mismatch for file %s (idx %s, %s vs. %s)" %
                    #      (r['name'], idx, rc[idx], sc[idx]))
                    stats['coverable_mismatch_count'] += 1

                    # Explicitly mark as not coverable
                    rc[idx] = -1
                if sc[idx] < 0 and rc[idx] >= 0:
                    rc[idx] = sc[idx]
                elif rc[idx] < 0 and sc[idx] >= 0:
                    pass
                elif rc[idx] >= 0 and sc[idx] >= 0:
                    rc[idx] += sc[idx]

    # Merge recursively
    merge_recursive(r, s)

    # Recursively re-calculate all summary fields
    calculate_summary_fields(r)

    return stats


def calculate_summary_fields(node, name=None):
    node["name"] = name
    node["linesTotal"] = 0
    node["linesCovered"] = 0

    if "children" in node:
        # This node has subtrees, recurse on them
        for child_name in node["children"]:
            child = node["children"][child_name]
            calculate_summary_fields(child, child_name)
            node["linesTotal"] += child["linesTotal"]
            node["linesCovered"] += child["linesCovered"]
    else:
        # This is a leaf, calculate linesTotal and linesCovered from
        # actual coverage data.
        coverage = node["coverage"]

        for line in coverage:
            if line >= 0:
                node["linesTotal"] += 1
                if line > 0:
                    node["linesCovered"] += 1

    # Calculate two more values based on total/covered because we need
    # them in the UI later anyway and can save some time by doing it here.
    node["linesMissed"] = node["linesTotal"] - node["linesCovered"]

    if node["linesTotal"] > 0:
        node["coveragePercent"] = round(((float(node["linesCovered"]) / node["linesTotal"]) * 100), 2)
    else:
        node["coveragePercent"] = 0.0


def apply_include_exclude_directives(node, directives):
    """
    Applies the given include and exclude directives to the given nodeself.

    Directives either start with a + or a - for include or exclude, followed
    by a colon and a regular expression. The regular expression must match the
    full path of the file(s) or dir(s) to include or exclude. All slashes in paths
    are forward slashes, must not have a trailing slash and special regex characters
    must be escaped by backslash.

    @param node: The coverage node to modify, in server-side recursive format
    @type node: dict

    @param directives: The directives to apply
    @type directives: list(str)

    This method modifies the node in-place, nothing is returned.

    IMPORTANT: This method does *not* recalculate any total/summary fields.
               You *must* call L{calculate_summary_fields} after applying
               this function one or more times to ensure correct results.
    """
    # Flatten all names in node
    from datetime import datetime
    s1 = datetime.now()
    names = get_flattened_names(node)
    s2 = datetime.now()
    print("get_flattened_names elapsed: %s" % (s2 - s1).total_seconds())

    names_to_remove = set()

    s1 = datetime.now()

    # Apply directives to names to determine name delection list
    for directive in directives:
        (what, regex) = directive.split(":", 2)

        if what == '+':
            cre = re.compile(regex)
            keep = set()
            for name in names_to_remove:
                if cre.match(name):
                    keep.add(name)

            # Remove any names we want to keep
            names_to_remove -= keep

            # We also need to remove any names that are prefix of what we want to keep
            prefix_keep = set()
            for name in names_to_remove:
                for keep_name in keep:
                    if keep_name.startswith(name + "/"):
                        prefix_keep.add(name)
            names_to_remove -= prefix_keep

        elif what == '-':
            if regex == '.*' or regex == '.+':
                # Special-case the exclude all pattern for performance
                names_to_remove.update(names)
            else:
                cre = re.compile(regex)
                for name in names:
                    if cre.match(name):
                        names_to_remove.add(name)
        else:
            raise RuntimeError("Unexpected directive prefix: %s" % what)

    s2 = datetime.now()
    print("applying elapsed: %s" % (s2 - s1).total_seconds())

    s1 = datetime.now()
    remove_names(node, names_to_remove)
    s2 = datetime.now()
    print("remove_names elapsed: %s" % (s2 - s1).total_seconds())


def remove_names(node, names):
    """
    Removes the given names (paths) from the given node.

    All slashes in paths are forward slashes and must not have a trailing
    slash.

    @param node: The coverage node to modify, in server-side recursive format
    @type node: dict

    @param names: The names (paths) to remove
    @type names: list(str)

    This method modifies the node in-place, nothing is returned.

    IMPORTANT: This method does *not* recalculate any total/summary fields.
               You *must* call L{calculate_summary_fields} after applying
               this function one or more times to ensure correct results.
    """
    def remove_name(node, name):
        current_node = node
        current_name = []
        current_name.extend(name[:-1])

        while current_name:
            if "children" not in current_node:
                return

            current_name_first = current_name.pop(0)

            if current_name_first not in current_node["children"]:
                return

            current_node = current_node["children"][current_name_first]

        if "children" not in current_node or name[-1] not in current_node["children"]:
            return

        del current_node["children"][name[-1]]

        def recurse_prune(node, name, idx):
            if "children" in node:
                if not node["children"]:
                    # Empty non-leaf
                    return True
                else:
                    if idx >= len(name) or name[idx] not in node["children"]:
                        return False

                    # Recurse. If we prune in recursion, check if our child is now empty
                    if recurse_prune(node["children"][name[idx]], name, idx + 1):
                        print("Pruning %s" % name[idx])
                        del node["children"][name[idx]]
                        if not node["children"]:
                            return True

            # We ended up with a non-empty non-leaf even after pruning, terminate algorithm
            return False

        # Perform a recursive prune along the path specified in name if we left an empty non-leaf
        if not current_node["children"]:
            recurse_prune(node, name, 0)
        return

    names = [name.split("/") for name in names]
    names.sort(key=len)

    for idx in range(0, len(names)):
        if idx >= len(names):
            return
        name = names[idx]
        remove_name(node, name)

        # Optimization: Delete all other names that this one is a prefix for
        for oidx in range(len(names) - 1, idx, -1):
            if name == names[oidx][:len(name)]:
                del names[oidx]


def get_flattened_names(node, prefix=""):
    """
    Returns a list of flattened paths (files and directories) of the given node.

    Paths will include the leading slash if the node a top-level node.
    All slashes in paths will be forward slashes and not use any trailing slashes.

    @param node: The coverage node to process, in server-side recursive format
    @type node: dict

    @param prefix: An optional prefix to prepend to each name
    @type prefix: str

    @return The list of all paths occurring in the given node.
    @rtype: list(str)
    """
    def __get_flattened_names(node, prefix, result):
        current_name = node["name"]
        if current_name is None:
            new_prefix = ""
        else:
            if prefix:
                new_prefix = "%s/%s" % (prefix, current_name)
            else:
                new_prefix = current_name
            result.add(new_prefix)

        if "children" in node:
            for child_name in node["children"]:
                child = node["children"][child_name]
                __get_flattened_names(child, new_prefix, result)
        return result

    return __get_flattened_names(node, prefix, set())
