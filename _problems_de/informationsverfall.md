---
title: Informationsverfall
description: Systemdokumentation wird über die Zeit veraltet, ungenau oder unvollständig,
  was sie unzuverlässig für Entscheidungsfindung und Systemverständnis macht.
category:
- Code
- Communication
related_problems:
- slug: poor-documentation
  similarity: 0.75
- slug: quality-degradation
  similarity: 0.7
- slug: unclear-documentation-ownership
  similarity: 0.65
- slug: information-fragmentation
  similarity: 0.65
- slug: rapid-system-changes
  similarity: 0.65
- slug: legacy-system-documentation-archaeology
  similarity: 0.65
solutions:
- architecture-decision-records
- documentation-as-code
- knowledge-sharing-practices
- audit-trail-management
- documentation-of-compatibility-requirements
- living-documentation
- timestamping
- written-first-communication
- knowledge-base
- code-reading-sessions
- application-portfolio-inventory
layout: problem
lang: de
en_slug: information-decay
---

## Description

Informationsverfall tritt auf, wenn Dokumentation, Spezifikationen und Wissensartefakte schrittweise veraltet, ungenau oder unvollständig werden, während sich Systeme weiterentwickeln. Dieser Verfall geschieht, weil Dokumentationspflege oft im Vergleich zur Feature-Entwicklung depriorisiert wird und der Aufwand, Informationen aktuell zu halten, unterschätzt wird. Während Informationen verfallen, verlieren Teams das Vertrauen in bestehende Dokumentation und greifen auf Stammeswissen oder Code-Archäologie zurück, was das System zunehmend schwer verständlich und wartbar macht.

## Indicators ⟡

- Die Dokumentation wurde trotz erheblicher Systemänderungen nicht aktualisiert
- Teammitglieder entdecken häufig, dass dokumentierte Verfahren nicht wie beschrieben funktionieren
- Neue Teammitglieder berichten, dass bestehende Dokumentation irreführend oder wenig hilfreich ist
- Code-Kommentare widersprechen dem tatsächlichen Systemverhalten
- Die API-Dokumentation stimmt nicht mit der aktuellen API-Funktionalität überein

## Symptoms ▲

- [Wissenssilos](wissenssilos.md)
<br/>  Wenn Dokumentation unzuverlässig wird, konzentriert sich Wissen bei erfahrenen Teammitgliedern statt in gemeinsam genutzten Artefakten.
- [Schwieriges Onboarding neuer Entwickler](schwieriges-onboarding-neuer-entwickler.md)
<br/>  Veraltete Dokumentation macht es neuen Teammitgliedern viel schwerer, das System zu lernen und produktiv zu werden.
- [Dokumentations-Archäologie bei Legacy-Systemen](dokumentations-archaeologie-bei-legacy-systemen.md)
<br/>  Verfallene Dokumentation zwingt Teams, Systemverhalten aus Code und Artefakten zurückzuentwickeln, statt zuverlässige Dokumente zu lesen.
- [Erhöhtes Risiko für Fehler](erhoehtes-risiko-fuer-fehler.md)
<br/>  Entwickler, die mit veralteter oder ungenauer Dokumentation arbeiten, treffen mit höherer Wahrscheinlichkeit falsche Annahmen und führen Fehler ein.
- [Langsame Feature-Entwicklung](langsame-feature-entwicklung.md)
<br/>  Entwickler verschwenden Zeit damit herauszufinden, wie das System tatsächlich funktioniert, weil die Dokumentation die Realität nicht mehr widerspiegelt.
- [Wissenslücken](wissensluecken.md)
<br/>  Wenn Dokumentation veraltet, entwickeln Teammitglieder Wissenslücken, weil sie nicht aus zuverlässigen schriftlichen Quellen lernen können.

## Causes ▼

- [Schlechte Dokumentation](schlechte-dokumentation.md)
<br/>  Systeme mit anfänglich schlechten Dokumentationspraktiken sind anfälliger für schnellen Informationsverfall.
- [Unklare Verantwortlichkeit für Dokumentation](unklare-verantwortlichkeit-fuer-dokumentation.md)
<br/>  Wenn niemand für die Pflege der Dokumentation verantwortlich ist, veraltet sie natürlich, während sich das System weiterentwickelt.
- [Zeitdruck](zeitdruck.md)
<br/>  Unter Lieferdruck werden Dokumentationsaktualisierungen zugunsten der Feature-Entwicklung depriorisiert, was den Verfall beschleunigt.
- [Hohe Fluktuation](hohe-fluktuation.md)
<br/>  Wenn kenntnisreiche Teammitglieder gehen, geht institutionelles Wissen über Dokumentationsgenauigkeit und -lücken verloren.

## Detection Methods ○

- **Dokumentations-Aktualitäts-Audit:** Nachverfolgung, wann Dokumentation zuletzt im Verhältnis zu Systemänderungen aktualisiert wurde
- **Genauigkeitsverifikation:** Testen dokumentierter Verfahren und Vergleich mit tatsächlichem Systemverhalten
- **Nutzerfeedback-Analyse:** Beobachtung von Beschwerden über ungenaue oder wenig hilfreiche Dokumentation
- **Bewertung der Onboarding-Erfahrung:** Bewertung des Erfolgs neuer Teammitglieder mit bestehender Dokumentation
- **Dokumentationsnutzungs-Tracking:** Beobachtung, welche Dokumentation aufgerufen wird und welche ignoriert wird
- **Identifikation von Wissenslücken:** Identifikation von Bereichen, in denen Systemwissen nur in den Köpfen von Personen existiert

## Examples

Ein Legacy-Finanzsystem hat umfassende Dokumentation, die während der ursprünglichen Implementierung vor fünf Jahren erstellt wurde, aber trotz zahlreicher Feature-Ergänzungen und architektonischer Änderungen nicht aktualisiert wurde. Neue Entwickler, die versuchen, das Zahlungsverarbeitungsmodul zu verstehen, stellen fest, dass dem dokumentierten Datenbankschema drei Tabellen und mehrere Spalten fehlen, die für regulatorische Compliance hinzugefügt wurden. Die API-Dokumentation zeigt Endpunkte, die nicht mehr existieren, und es fehlt Dokumentation für neue Authentifizierungsanforderungen. Wenn Probleme in Produktion auftreten, müssen Entwickler das aktuelle Systemverhalten zurückentwickeln, statt sich auf Dokumentation zu verlassen, was die Fehlerbehebungszeit erheblich verlängert. Ein weiteres Beispiel betrifft eine Microservices-Plattform, bei der die Servicearchitektur-Dokumentation das ursprüngliche Design mit sechs Diensten zeigt, aber das System sich zu zwölf Diensten mit komplexen gegenseitigen Abhängigkeiten weiterentwickelt hat. Die Deployment-Dokumentation referenziert immer noch den alten Containerisierungsansatz und erwähnt nicht das aktuelle Kubernetes-Setup, was es neuen Teammitgliedern unmöglich macht, die Anwendung erfolgreich zu deployen.
