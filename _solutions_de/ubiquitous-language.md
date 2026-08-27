---
title: Ubiquitous Language
description: Ausrichtung des Vokabulars von Entwicklern und Fachexperten
  in Code und Gespräch.
category:
- Communication
- Code
problems:
- stakeholder-developer-communication-gap
- poor-domain-model
- difficult-code-comprehension
- requirements-ambiguity
- poor-naming-conventions
- inconsistent-naming-conventions
- knowledge-gaps
- misaligned-deliverables
- communication-risk-within-project
- language-barriers
- difficult-to-understand-code
- custom-report-sprawl
- master-data-ownership-gaps
layout: solution
lang: de
en_slug: ubiquitous-language
related_solutions:
- slug: consistent-terminology
  similarity: 0.85
- slug: plain-language
  similarity: 0.75
- slug: domain-modeling
  similarity: 0.75
- slug: domain-patterns
  similarity: 0.75
- slug: domain-specific-languages
  similarity: 0.7
- slug: pattern-language
  similarity: 0.7
---

## Description

Ubiquitous Language ist die bewusste Praxis, ein einziges, konsistentes Vokabular für Domänenkonzepte über Gespräche, Dokumentation, Code, Datenbankschemata und API-Verträge hinweg zu nutzen, sodass ein Begriff exakt dasselbe bedeutet, egal wer ihn nutzt oder wo er erscheint. Es wird etabliert, indem die Wörter verglichen werden, die Geschäfts-Stakeholder tatsächlich nutzen, mit den Wörtern, die in der Codebasis vorhanden sind, und dann die Lücken geschlossen werden — kryptische technische Bezeichner werden während der Refaktorierung in Domänenbegriffe umbenannt, und Fälle werden gelöst, in denen verschiedene Teams still unterschiedliche Wörter für dasselbe Konzept übernommen haben. In Legacy-Systemen ist diese Lücke oft ungewöhnlich breit, weil der Code häufig nach Datenbankspalten-Längenbeschränkungen, Entwickler-Abkürzungen oder technischen Konventionen aus Jahrzehnten benannt wurde, von denen keine irgendeine Verpflichtung hatte, mitzuverfolgen, wie das Geschäft selbst über seine eigene Domäne spricht, und die Personen, die die ursprünglichen Namenswahlen hätten erklären können, sind typischerweise längst gegangen. Die Diskrepanz ist nicht nur kosmetisch: Sie ist eine direkte Quelle für missverständnisgetriebene Fehler und Nacharbeit, da Entwickler und Fachexperten, die still aneinander vorbei über dasselbe zugrunde liegende Konzept reden, dazu neigen, mit vollem Vertrauen das Falsche zu bauen. Ein gemeinsames Glossar zu etablieren und es durch Code-Umbenennungen, Reviews und alltägliche Kommunikation durchzusetzen macht Legacy-Code für Neuankömmlinge erheblich verständlicher und lässt Fachexperten sinnvoll an technischen Diskussionen teilnehmen, von denen sie sonst allein durch Vokabular ausgeschlossen wären.

## How to Apply ◆

> In Legacy-Systemen ist die Lücke zwischen Domänensprache und Codesprache oft jahrzehntebreit — sie durch Ubiquitous Language zu überbrücken macht die Codebasis sowohl für Entwickler als auch Fachexperten verständlich.

- Erstellen Sie ein Glossar von Domänenbegriffen, indem Sie Geschäfts-Stakeholder interviewen und ihr Vokabular mit den in der Legacy-Codebasis genutzten Begriffen vergleichen — die Diskrepanzen offenbaren, wo Missverständnisse am wahrscheinlichsten sind.
- Benennen Sie Code-Elemente (Klassen, Methoden, Variablen, Datenbankspalten) während der Refaktorierung um, um Domänenterminologie zu nutzen, und beseitigen Sie kryptische Abkürzungen und technischen Jargon, den nur die ursprünglichen Entwickler verstanden.
- Stellen Sie sicher, dass derselbe Begriff überall dasselbe bedeutet — in Gesprächen, Dokumentation, Code, Datenbankschemata und API-Verträgen — und lösen Sie explizit Fälle, in denen verschiedene Teams unterschiedliche Wörter für dasselbe Konzept nutzen.
- Nutzen Sie die Ubiquitous Language in aller Teamkommunikation, einschließlich Commit-Nachrichten, Pull-Request-Beschreibungen und Architecture Decision Records.
- Wenn Fachexperten einen Begriff nutzen, der im Code nicht existiert, untersuchen Sie, ob das Konzept im Modell fehlt oder einfach anders benannt ist.
- Überarbeiten und entwickeln Sie die Sprache weiter, während sich das Domänenverständnis während der Modernisierung vertieft — der erste Satz von Begriffen ist selten der endgültige.

## Tradeoffs ⇄

> Ubiquitous Language reduziert Missverständnisse und verbessert die Codelesbarkeit, erfordert aber anhaltende Disziplin und Bereitschaft, etablierte Code-Elemente umzubenennen.

**Vorteile:**

- Beseitigt eine große Quelle von Fehlern und Nacharbeit, die durch Entwickler und Fachexperten verursacht wird, die unterschiedliche Begriffe für dasselbe Konzept oder denselben Begriff für unterschiedliche Konzepte nutzen.
- Macht Legacy-Code verständlicher, indem kryptische Abkürzungen durch bedeutungsvolle Domänenbegriffe ersetzt werden.
- Ermöglicht Fachexperten, sinnvoll an Code-Reviews und Designdiskussionen teilzunehmen.
- Reduziert die Einarbeitungszeit für neue Entwickler, die die Codebasis durch Lesen ihrer domänenausgerichteten Namen verstehen können.

**Kosten und Risiken:**

- Das Umbenennen etablierter Code-Elemente in einem Legacy-System kann weitreichende Änderungen auslösen und erfordert sorgfältige Refaktorierung mit guter Testabdeckung.
- Fachexperten könnten selbst inkonsistente Terminologie nutzen, was moderierte Diskussionen zur Lösung von Konflikten erfordert.
- Manche technischen Konzepte (Caches, Warteschlangen, Verbindungspools) haben kein natürliches Domänenäquivalent und sollten ihre technischen Namen behalten.
- Die Aufrechterhaltung von Sprachkonsistenz über ein großes Team hinweg erfordert laufende Wachsamkeit und könnte ein lebendes Glossar brauchen, das jemand besitzt.

## How It Could Be

> Das folgende Szenario veranschaulicht die Wirkung der Etablierung von Ubiquitous Language während der Legacy-Modernisierung.

Das Legacy-System eines Gewerbeimmobilienunternehmens nutzte Abkürzungen aus seinem Datenbankdesign der 1990er-Jahre: `PROP_UNIT` für vermietbare Flächen, `TNT_REC` für Mieterdatensätze, `OCC_PCT` für Belegungsraten, und `LSE_TERM` für Mietverträge. Neu zum Team stoßende Entwickler verbrachten Wochen damit, dieses private Vokabular zu lernen, und Anforderungsdiskussionen wurden ständig durch Übersetzungsverwirrung entgleist — wenn ein Immobilienverwalter "Suite" sagte, hörten die Entwickler "Unit", und wenn die Datenbank `LSE_TERM` sagte, konnte es entweder das Mietdokument oder die Mietdauer bedeuten. Während der Modernisierung etablierte das Team ein gemeinsames Glossar, das Begriffe aus der Immobilienverwaltungsbranche mit Codenamen abglich: `LeasableSpace`, `Tenant`, `OccupancyRate`, `LeaseAgreement`. Die Umbenennungsanstrengung berührte Hunderte von Dateien, reduzierte aber sofort die Rate der Anforderungsmissverständnisse. Neue Entwickler berichteten, zwei Wochen schneller produktiv zu sein als ihre Vorgänger, und Immobilienverwalter konnten nun API-Dokumentation ohne einen Übersetzungsleitfaden lesen.
