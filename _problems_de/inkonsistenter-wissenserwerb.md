---
title: Inkonsistenter Wissenserwerb
description: Neue Teammitglieder lernen unterschiedliche Aspekte und Tiefen von Systemwissen,
  abhängig von ihrem Mentor oder Lernpfad, was zu ungleicher Kompetenzverteilung
  führt.
category:
- Communication
- Process
- Team
related_problems:
- slug: inconsistent-onboarding-experience
  similarity: 0.75
- slug: knowledge-gaps
  similarity: 0.65
- slug: skill-development-gaps
  similarity: 0.65
- slug: knowledge-silos
  similarity: 0.65
- slug: uneven-workload-distribution
  similarity: 0.65
- slug: inconsistent-quality
  similarity: 0.65
solutions:
- knowledge-sharing-practices
- pair-and-mob-programming
- domain-quiz
- knowledge-rotation
- structured-onboarding-program
- code-reading-sessions
- internal-technical-coaching
- knowledge-base
- documentation-as-code
- communities-of-practice
layout: problem
lang: de
en_slug: inconsistent-knowledge-acquisition
---

## Description

Inkonsistenter Wissenserwerb tritt auf, wenn neue Teammitglieder unterschiedliche Arten, Tiefen oder Qualitäten von Wissen erhalten, abhängig davon, wer sie betreut, welche Ressourcen sie nutzen oder welche Teile des Systems sie zuerst kennenlernen. Dies führt zu ungleicher Kompetenzverteilung über das Team hinweg, wobei manche Entwickler zu Experten in bestimmten Bereichen werden, während sie mit anderen völlig unvertraut bleiben, selbst nach Monaten der Arbeit.

## Indicators ⟡

- Neue Mitarbeiter mit ähnlichem Hintergrund und Erfahrungsniveau entwickeln sehr unterschiedliche Kompetenzen
- Teammitglieder haben ein völlig unterschiedliches Verständnis derselben Systemkomponenten
- Manche Entwickler können bestimmte Arten von Aufgaben handhaben, während andere es trotz ähnlicher Betriebszugehörigkeit nicht können
- Wissenslücken erscheinen zufällig über das Team verteilt, statt Erfahrungsniveaus zu folgen
- Schulungsergebnisse variieren erheblich, abhängig davon, wer die Schulung durchführt

## Symptoms ▲

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Teammitglieder unterschiedliche Aspekte des Systems lernen, wird Wissen fragmentiert und bei Einzelpersonen isoliert.
- [Unpassendes Fähigkeitsprofil](unpassendes-faehigkeitsprofil.md)
<br/>  Ungleiche Lernpfade lassen Teammitglieder mit Fähigkeitslücken zurück, die nicht zu ihren zugewiesenen Verantwortlichkeiten passen.
- [Wissensabhängigkeit](wissensabhaengigkeit.md)
<br/>  Weil jede Person nur bestimmte Aspekte lernte, bleiben Teammitglieder von anderen für Wissen abhängig, das sie nie erworben haben.
- [Engpassbildung](engpassbildung.md)
<br/>  Nur bestimmte Personen können bestimmte Aufgaben handhaben, weil Wissen während des Erwerbs ungleich verteilt wurde.
- [Ungleichmäßige Arbeitslastverteilung](ungleichmaessige-arbeitslastverteilung.md)
<br/>  Aufgaben werden zugewiesen, basierend darauf, wer was weiß, statt auf Verfügbarkeit, was unausgeglichene Arbeitslasten schafft.

## Causes ▼

- [Inkonsistente Onboarding-Erfahrung](inkonsistente-onboarding-erfahrung.md)
<br/>  Unterschiedliche Onboarding-Erfahrungen geben neuen Mitarbeitern unterschiedliche Ausgangspunkte für den Wissenserwerb.
- [Unzureichende Mentoring-Struktur](unzureichende-mentoring-struktur.md)
<br/>  Ohne einen systematischen Mentoring-Ansatz hängt das, was neue Mitarbeiter lernen, stark von der Expertise und dem Stil ihres individuellen Mentors ab.
- [Zusammenbruch des Wissensaustauschs](zusammenbruch-des-wissensaustauschs.md)
<br/>  Ineffektiver Wissensaustausch bedeutet, dass neue Mitarbeiter ihr mentorabhängiges Lernen nicht mit breiterem Teamwissen ergänzen können.

## Detection Methods ○

- **Wissens-Mapping:** Befragung von Teammitgliedern zur Identifikation, was jede Person über unterschiedliche Systembereiche weiß und nicht weiß
- **Muster der Aufgabenzuweisung:** Analyse, welchen Teammitgliedern welche Arten von Aufgaben zugewiesen werden und warum
- **Wirksamkeit von Cross-Training:** Testen, ob Teammitglieder an Aufgaben außerhalb ihrer anfänglichen Fokusbereiche arbeiten können
- **Vergleich der Onboarding-Ergebnisse:** Vergleich von Wissen und Fähigkeiten, die unterschiedliche neue Mitarbeiter nach ähnlichen Zeiträumen erworben haben
- **Analyse der Mentor-Wirkung:** Bewertung, wie unterschiedliche Mentoren die Lernergebnisse neuer Mitarbeiter beeinflussen

## Examples

Drei Entwickler treten innerhalb eines Monats einem Fintech-Team bei. Der erste Entwickler wird vom Architekten betreut und lernt über Systemdesign, Datenfluss und Integrationsmuster, weiß aber wenig über die Geschäftsdomäne. Der zweite wird mit einem Fachexperten gepaart und wird versiert in Finanzvorschriften und Geschäftsregeln, kämpft aber mit technischen Implementierungsdetails. Der dritte Entwickler arbeitet hauptsächlich an Fehlerbehebungen und lernt Debugging-Techniken und Navigation in Legacy-Code, hat aber begrenztes Verständnis sowohl von Architektur als auch von Geschäftslogik. Nach sechs Monaten kann keiner von ihnen effektiv an komplexen Features zusammenarbeiten, weil jeder tiefes Wissen in unterschiedlichen Bereichen mit minimaler Überschneidung hat. Ein weiteres Beispiel betrifft eine E-Commerce-Plattform, bei der das Lernen neuer Entwickler vollständig davon abhängt, welchem Team sie anfänglich zugewiesen werden – diejenigen, die mit dem Checkout-Team beginnen, lernen Zahlungsabwicklung gründlich, wissen aber nichts über Bestandsverwaltung, während diejenigen, die mit dem Katalog-Team beginnen, Produktdaten verstehen, aber Probleme bei der Bestellverarbeitung nicht beheben können.
