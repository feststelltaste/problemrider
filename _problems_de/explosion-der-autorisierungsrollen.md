---
title: Explosion der Autorisierungsrollen
description: Das Rollen- und Berechtigungsmodell ist auf Tausende von Einträgen
  angewachsen, die sich nur anhäufen, sodass niemand mehr sagen kann, wer was tun darf.
category:
- Security
- Operations
- Process
related_problems:
- slug: custom-report-sprawl
  similarity: 0.6
- slug: authorization-flaws
  similarity: 0.6
- slug: excessive-customization
  similarity: 0.55
- slug: low-code-customization-sprawl
  similarity: 0.55
- slug: change-management-chaos
  similarity: 0.5
- slug: approval-dependencies
  similarity: 0.5
solutions:
- role-model-rationalization
- authorization-concept
- role-based-access-control
- least-privilege
- domain-based-authorization-concept
- clear-ownership-model
- customization-under-version-control
- security-audits
- attribute-usage-analysis
- quality-ratchet
layout: problem
lang: de
en_slug: authorization-role-explosion
---

## Description

Die Explosion der Autorisierungsrollen entsteht, wenn das Berechtigungsmodell eines kommerziell erworbenen Softwaresystems kontinuierlich wächst und nie verkleinert wird. Jede neue Anforderung erzeugt eine neue Rolle statt einer Änderung an einer bestehenden, weil das Ändern einer bestehenden Rolle riskiert, Zugriff zu entfernen, auf den sich jemand verlässt, und niemand feststellen kann, wer das ist. Rollen werden für Einzelpersonen kopiert, häufen bei Stellenwechseln Berechtigungen an und überleben die Positionen, für die sie geschaffen wurden. Das Ergebnis ist ein Modell, das die Organisation nicht mehr beschreibt: Es ist eine sedimentäre Aufzeichnung jeder jemals gewährten Zugriffsanfrage. Die praktischen Konsequenzen sind, dass niemand beantworten kann, wer eine sensible Aktion durchführen kann, Zugriffsüberprüfungen unmöglich sinnvoll durchzuführen sind und jedes Audit Befunde produziert, die durch das Hinzufügen weiterer Rollen behoben werden.

## Indicators ⟡

- Die Anzahl der Rollen liegt nahe an der Anzahl der Nutzer oder übersteigt sie
- Rollen tragen Namen wie den Namen einer Person, ein beendetes Projekt oder eine Versionsnummer
- Neuer Zugriff wird gewährt, indem die Rollen eines bestehenden Nutzers kopiert werden, statt eine definierte Rolle zuzuweisen
- Zugriffsüberprüfungen werden durchgeführt, indem Manager gebeten werden, Listen zu genehmigen, die sie nicht sinnvoll bewerten können
- Niemand kann innerhalb eines Tages beantworten, wer eine bestimmte sensible Transaktion durchführen kann
- Rollen werden nur hinzugefügt; es gibt keinen Prozess, durch den eine Rolle entfernt wird
- Zugriffsprobleme werden gelöst, indem eine Berechtigung hinzugefügt wird, nie indem untersucht wird, warum die bestehende unzureichend war

## Symptoms ▲

- [Autorisierungsfehler](autorisierungsfehler.md)
<br/>  Angehäufte Berechtigungen erzeugen Kombinationen, die niemand beabsichtigt hat, einschließlich Zugriff, der Anforderungen der Funktionstrennung verletzt.
- [Regulatorische Compliance-Drift](regulatorische-compliance-drift.md)
<br/>  Zugriffskontrollen, die nicht beschrieben werden können, können einem Prüfer gegenüber nicht nachgewiesen werden, unabhängig davon, ob sie in der Praxis angemessen sind.
- [Langsame Vorfallslösung](langsame-vorfallsloesung.md)
<br/>  Um festzustellen, ob eine Aktion erlaubt war und von wem, muss ein effektives Berechtigungsset über viele überlappende Rollen hinweg rekonstruiert werden.
- [Erhöhte manuelle Arbeit](erhoehte-manuelle-arbeit.md)
<br/>  Das Bereitstellen, Überprüfen und Korrigieren von Zugriff verbraucht kontinuierlichen administrativen Aufwand, der mit der Größe des Modells wächst.
- [Unsichtbarkeit technischer Schulden](unsichtbarkeit-technischer-schulden.md)
<br/>  Das Berechtigungsmodell wird selten überhaupt als Schuld gezählt, sodass seine Anhäufung nicht gemeldet und ihre Kosten niemandem zugeschrieben werden.
- [Nutzerfrustration](nutzerfrustration.md)
<br/>  Nutzer werden durch fehlende Berechtigungen blockiert und andernorts übermäßig berechtigt, und der Lösungszyklus für beides ist langsam.

## Causes ▼

- [Übermäßige Anpassung](uebermaessige-anpassung.md)
<br/>  Anpassung vervielfacht die Transaktionen und Objekte, die eine Autorisierung erfordern, und das Berechtigungsmodell wächst mit ihnen.
- [Angst vor Breaking Changes](angst-vor-breaking-changes.md)
<br/>  Das Entfernen einer Berechtigung könnte jemanden blockieren, und da niemand feststellen kann, wen, ist die sichere Aktion immer, hinzuzufügen statt zu ändern.
- [Anpassungen außerhalb der Versionskontrolle](anpassungen-ausserhalb-der-versionskontrolle.md)
<br/>  Rollendefinitionen, die keine Historie und keine Urheberschaft tragen, können nicht überprüft werden, sodass ihr Wachstum unbeobachtet bleibt.
- [Fehlende Eigenverantwortung und Rechenschaftspflicht](fehlende-eigenverantwortung-und-rechenschaftspflicht.md)
<br/>  Ohne einen Verantwortlichen für das Autorisierungsmodell als Ganzes ist niemand für seine Kohärenz zuständig, und jeder ist nur für seine eigenen Anfragen zuständig.
- [Kurzfristiger Fokus](kurzfristiger-fokus.md)
<br/>  Das Gewähren des angeforderten Zugriffs löst die heutige Blockade; das Neugestalten des Modells tut das nicht und wird daher nie gewählt.
- [Unzureichende Anforderungserhebung](unzureichende-anforderungserhebung.md)
<br/>  Zugriff wird danach angefragt, was eine bestimmte Person heute tun muss, statt danach, welche Rolle die Organisation tatsächlich hat.

## Detection Methods ○

- Rollen gegen Nutzer und gegen Funktionsbereiche zählen; eine Rollenanzahl, die die Anzahl unterschiedlicher Funktionsbereiche um eine Größenordnung übersteigt, ist diagnostisch
- Rollen identifizieren, die keinem Nutzer, genau einem Nutzer zugewiesen oder seit mehreren Jahren unverändert sind
- Effektive Berechtigungen für eine Stichprobe von Nutzern berechnen und mit den Anforderungen ihrer Funktion vergleichen
- Das Modell mit einer konkreten Frage testen – wer kann eine Zahlung über einem Schwellenwert genehmigen – und die Antwortzeit messen
- Prüfen, ob jemals eine Rolle entfernt wurde und welcher Prozess dafür verwendet würde
- Nach Rollennamen suchen, die Personennamen, Projektnamen oder Datumsangaben enthalten

## Examples

Eine ERP-Installation mit 2.400 Nutzern hatte 3.100 Rollen. Die Analyse ergab, dass 890 genau einem Nutzer zugewiesen waren, 400 niemandem, und dass die größte einzelne Kategorie aus Rollen bestand, die durch Kopieren der Rollen eines anderen Nutzers beim Onboarding erstellt und dann geändert wurden. Von einem Prüfer gefragt, wer eine Buchung über einer Wesentlichkeitsschwelle vornehmen könne, brauchte das Team neun Tage, um eine Antwort zu erstellen, und die Antwort identifizierte 34 Nutzer, von denen der Finanzdirektor 19 als angemessen erkannte. Die verbleibenden 15 hatten die Berechtigung durch Rollenkombinationen angehäuft, die für andere Zwecke geschaffen wurden, in drei Fällen durch eine Rolle, die nach einem 2017 abgeschlossenen Projekt benannt war.

Der Anhäufungsmechanismus zeigte sich darin, wie ein typischer Fall entstand. Ein Nutzer wechselte vom Einkauf in die Finanzabteilung. Seine bestehenden Rollen wurden beibehalten, weil das Entfernen etwas hätte kaputt machen können und niemand feststellen konnte, was davon abhing, und Finanzrollen wurden hinzugefügt. Über elf Jahre hatte die Organisation etwa 4.000 solcher Wechsel bearbeitet. Das Berechtigungsmodell enthielt keinen Mechanismus, durch den jemals etwas weggenommen wurde, und jeder Audit-Befund war historisch abgeschlossen worden, indem eine restriktivere Rolle geschaffen und zusätzlich zu dem bereits Vorhandenen zugewiesen wurde.
