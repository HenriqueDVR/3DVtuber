"""Whose certificates a download believes. Plain Python, standard library only.

Python on Windows trusts everything in Windows' certificate stores, the intermediate ones ("CA")
with the roots. Windows keeps intermediates it once met there for good, and an expired one is then
taken in place of the fresh one the server sends: blender.org, with a certificate good until
December, was refused as "certificate has expired" on 2026-10-07 (a 2025 copy of Let's Encrypt's
"ISRG Root X2" in the store), while Windows itself, the browsers and Blender (its extension
downloads) opened it. The roots alone, and the chain the server sends, is what every other
system checks against.
"""
import ssl
import sys
import urllib.error
import urllib.request


def roots():
    """A context that checks a server against Windows' root authorities alone (elsewhere: the
    system's own list, as ever). Nothing is trusted that Windows doesn't trust."""
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    if sys.platform != "win32":
        context.load_default_certs()
        return context
    for certificate, encoding, trust in ssl.enum_certificates("ROOT"):
        if encoding == "x509_asn" and (trust is True or ssl.Purpose.SERVER_AUTH.oid in trust):
            try:
                context.load_verify_locations(cadata=certificate)
            except ssl.SSLError:
                pass  # one unreadable certificate in the store is not a reason to trust none
    return context


def open_url(request, timeout=None, first=urllib.request.urlopen):
    """urlopen; a certificate refused on Windows is asked for once more against the roots alone."""
    try:
        return first(request, timeout=timeout)
    except urllib.error.URLError as error:
        if sys.platform != "win32" or not isinstance(error.reason, ssl.SSLCertVerificationError):
            raise
    return urllib.request.urlopen(request, timeout=timeout, context=roots())

