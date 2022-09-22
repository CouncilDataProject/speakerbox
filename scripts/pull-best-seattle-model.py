#!/usr/bin/env python
# -*- coding: utf-8 -*-

from quilt3 import Package

###############################################################################


package = Package.browse(
    "speakerbox/model",
    "s3://evamaxfield-uw-equitensors-speakerbox",
    "453d51cc7006d2ba26640ba91eed67a5f8a9315d7c25d95f81072edb20054054",
)
package["trained-speakerbox"].fetch("seattle-best-model")
