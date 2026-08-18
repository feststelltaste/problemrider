---
title: Schlechte Dokumentation
description: Systemdokumentation ist veraltet, unvollständig, ungenau oder schwer
  zu finden und effektiv zu nutzen.
category:
- Code
- Communication
related_problems:
- slug: unclear-documentation-ownership
  similarity: 0.75
- slug: information-decay
  similarity: 0.75
- slug: legacy-system-documentation-archaeology
  similarity: 0.65
- slug: information-fragmentation
  similarity: 0.6
- slug: rapid-system-changes
  similarity: 0.6
- slug: difficult-developer-onboarding
  similarity: 0.6
solutions:
- architecture-decision-records
- definition-of-done
- documentation-as-code
- api-documentation
- architecture-documentation
- checklists
- code-comments
- consistent-terminology
- contextual-help
- documentation-of-compatibility-requirements
- frequently-asked-questions-faq
- living-documentation
- portability-checklists
- prepared-statements
- privacy-by-design
- security-certification
- security-frameworks
- security-policies-for-development
- timestamping
- fair-source
- knowledge-base
- plain-language
- runbooks
- video-tutorials
layout: problem
lang: de
en_slug: poor-documentation
---

## Description

Schlechte Dokumentation tritt auf, wenn die schriftlichen Informationen über ein System, seine Architektur, Geschäftsregeln, APIs, Deployment-Verfahren und operative Anforderungen unzureichend sind, damit Entwickler das System effektiv verstehen und mit ihm arbeiten können. Dies umfasst Dokumentation, die veraltet, unvollständig, ungenau, schlecht organisiert oder schlicht nicht vorhanden ist. Schlechte Dokumentation zwingt Entwickler, sich auf Stammeswissen und Experimentieren zu verlassen, was die Entwicklung verlangsamt und das Fehlerrisiko erhöht.

## Indicators ⟡

- Dokumentation wurde seit Monaten oder Jahren nicht aktualisiert, trotz Systemänderungen
- Entwickler konsultieren bestehende Dokumentation selten, weil bekannt ist, dass sie unzuverlässig ist
- Neue Teammitglieder können ohne umfangreiche Eins-zu-eins-Anleitung nicht beginnen
- API-Dokumentation entspricht nicht dem tatsächlichen API-Verhalten
- Deployment- und Betriebsverfahren existieren nur als Stammeswissen

## Symptoms ▲

- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Neue Entwickler können sich nicht unabhängig einarbeiten, wenn Dokumentation veraltet oder fehlend ist, was umfangreiche Betreuung erfordert.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Ohne zuverlässige Dokumentation werden Teams abhängig von bestimmten Personen, die Systemwissen in ihren Köpfen halten.
- [Langsamer Wissenstransfer](langsamer-wissenstransfer.md)
<br/>  Schlechte Dokumentation zwingt Wissenstransfer dazu, durch langsame Eins-zu-eins-Gespräche statt Selbstbedienungs-Lektüre zu erfolgen.
- [Annahmenbasierte Entwicklung](annahmenbasierte-entwicklung.md)
<br/>  Wenn Dokumentation bestehender Geschäftsregeln oder Systemverhaltens fehlt oder unzuverlässig ist, raten Entwickler möglicherweise bei diesem spezifischen Verhalten, statt es nachzuschlagen; die meisten anderen annahmenbasierten Entwicklungen entstehen jedoch eher aus Lücken bei der Anforderungserhebung oder Stakeholder-Kommunikation als aus allgemeiner Dokumentationsqualität.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Ohne akkurate Dokumentation von Geschäftsregeln und Systemverhalten führen Entwickler mit höherer Wahrscheinlichkeit Fehler durch Missverständnisse ein.
- [Verlängerte Recherchezeit](verlaengerte-recherchezeit.md)
<br/>  Entwickler verbringen exzessive Zeit damit, Systemverhalten zurückzuentwickeln, das dokumentiert sein sollte.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Ohne Dokumentation müssen Entwickler die Codebasis zurückentwickeln, bevor sie Features hinzufügen können, was die Lieferung verlangsamt.
- [Langsame Entwicklungsgeschwindigkeit](langsame-entwicklungsgeschwindigkeit.md)
<br/>  Zeit, die für das Entziffern undokumentierter Systeme aufgewendet wird, summiert sich über das Team hinweg und verlangsamt das gesamte Liefertempo über jede einzelne Aufgabe hinaus.

## Causes ▼

- [Termindruck](termindruck.md)
<br/>  Unter Zeitdruck ist Dokumentation die erste Aktivität, die gestrichen wird, was zu wachsenden Dokumentationslücken führt.
- [Unklare Verantwortlichkeit für Dokumentation](unklare-verantwortlichkeit-fuer-dokumentation.md)
<br/>  Wenn niemand dafür verantwortlich ist, Dokumentation aktuell zu halten, verfällt sie über die Zeit, während sich das System weiterentwickelt.
- [Schnelle Systemänderungen](schnelle-systemaenderungen.md)
<br/>  Häufige Systemänderungen überholen Dokumentations-Updates, was sie schnell veralten lässt.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Teams, die sich auf kurzfristige Lieferung fokussieren, priorisieren die Dokumentationspflege als langfristige Investition ab.

## Detection Methods ○

- **Dokumentations-Aktualitätsanalyse:** Vergleich von Dokumentationsdaten mit aktuellen Systemänderungen
- **Dokumentationsnutzungsverfolgung:** Überwachung, wie oft Teammitglieder tatsächlich bestehende Dokumentation nutzen
- **Dokumentationslückenbewertung:** Identifikation von Bereichen, in denen Dokumentation fehlt oder unzureichend ist
- **Feedback neuer Mitarbeiter zur Dokumentation:** Sammlung von Feedback von neuen Teammitgliedern zur Effektivität der Dokumentation
- **Dokumentationsgenauigkeits-Audit:** Verifikation, dass bestehende Dokumentation dem tatsächlichen Systemverhalten entspricht

## Examples

Eine Microservices-Architektur hat 47 verschiedene Services, aber nur 12 haben überhaupt API-Dokumentation, und die meiste dieser Dokumentation wurde zuletzt vor 18 Monaten aktualisiert. Neue Entwickler, die Service-Interaktionen verstehen wollen, müssen API-Verträge durch Lesen von Code und manuelles Testen von Endpunkten zurückentwickeln. Die Deployment-Dokumentation referenziert Server, die vor zwei Jahren stillgelegt wurden, und der tatsächliche Deployment-Prozess umfasst eine Reihe manueller Schritte, die nur im Gedächtnis zweier Senior-Entwickler existieren. Wenn diese Entwickler in den Urlaub gehen, stoppen Deployments entweder komplett oder scheitern, weil niemand sonst den vollständigen Prozess kennt. Ein weiteres Beispiel betrifft ein Finanzhandelssystem, bei dem die Geschäftsregeldokumentation während der ursprünglichen Implementierung vor fünf Jahren geschrieben wurde, aber trotz Dutzender regulatorischer Änderungen und Modifikationen der Geschäftsanforderungen nicht aktualisiert wurde. Entwickler, die neue Features implementieren, müssen Geschäftsanalysten befragen und Legacy-Code untersuchen, um aktuelle Anforderungen zu verstehen, wobei sie oft undokumentierte Ausnahmen und Sonderfälle entdecken, die Fehler in Produktion verursachen.
