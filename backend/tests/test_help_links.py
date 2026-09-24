"""Help-center "Learn more" links: the KB stores a bare SLUG per chunk and the
full, language-correct URL is assembled at answer time. These tests lock that
contract (slug in → localized URL out) end to end, offline."""
import uuid

import pytest

import parser as kb_parser
import qa


# --------------------------------------------------------------------------- #
# parser: marker line → bare slug
# --------------------------------------------------------------------------- #
def test_help_slug_normalises_full_url_and_bare_slug():
    assert kb_parser._help_slug("settings-general-information") == "settings-general-information"
    # legacy full URL → last path segment
    assert kb_parser._help_slug("https://dev.1audit.com/en/app/knowledge_base/helps/users") == "users"
    # trailing punctuation stripped
    assert kb_parser._help_slug("branches.") == "branches"


def test_help_url_re_matches_en_and_ar_markers():
    m_en = kb_parser._HELP_URL_RE.search("Help page: settings-general-information")
    m_ar = kb_parser._HELP_URL_RE.search("صفحة المساعدة: organization")
    assert m_en and kb_parser._help_slug(m_en.group(1)) == "settings-general-information"
    assert m_ar and kb_parser._help_slug(m_ar.group(1)) == "organization"


# --------------------------------------------------------------------------- #
# qa: slug + language → full URL
# --------------------------------------------------------------------------- #
def test_build_help_url_is_language_prefixed():
    en = qa.build_help_url("clients", "en")
    ar = qa.build_help_url("clients", "ar")
    assert en.endswith("/en/app/knowledge_base/helps/clients")
    assert ar.endswith("/ar/app/knowledge_base/helps/clients")
    # same slug, only the language segment differs
    assert en.replace("/en/", "/LANG/") == ar.replace("/ar/", "/LANG/")


def test_build_help_url_none_when_no_slug():
    assert qa.build_help_url(None, "en") is None
    assert qa.build_help_url("", "ar") is None


def test_build_help_url_defaults_unknown_language_to_en():
    assert "/en/" in qa.build_help_url("sales", "fr")


# --------------------------------------------------------------------------- #
# prompt: the excerpt carries the localized URL, chosen by answer language
# --------------------------------------------------------------------------- #
def _chunk(slug):
    return qa.RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=uuid.uuid4(),
        document_filename="kb.docx",
        content="To create a client, open Clients and click Add new client.",
        similarity=0.9,
        section_heading="Clients",
        help_url=slug,
    )


def test_prompt_embeds_localized_help_url_per_language():
    chunks = [_chunk("clients")]
    prompt_en = qa._build_user_prompt("how to create client?", chunks, [], language="en")
    prompt_ar = qa._build_user_prompt("كيفية إنشاء عميل؟", chunks, [], language="ar")
    assert 'help_url="https://dev.1audit.com/en/app/knowledge_base/helps/clients"' in prompt_en
    assert 'help_url="https://dev.1audit.com/ar/app/knowledge_base/helps/clients"' in prompt_ar


def test_prompt_has_no_help_attr_when_slug_missing():
    chunks = [_chunk(None)]
    prompt = qa._build_user_prompt("q", chunks, [], language="en")
    assert "help_url=" not in prompt
